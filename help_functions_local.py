import psycopg2
import torch
import nibabel as nib
from pydicom.filebase import DicomBytesIO
import re
import numpy as np
import sys
import skimage
import skimage.io
import skimage.transform
import pydicom
import os
import scipy
from datetime import datetime
import time
import timeit
from collections import Counter
import boto3
import json
import SimpleITK as sitk
import ukbb_cardiac.common.cardiac_utils_edited_D3PO as card_utils
CARDUTILS_IMPORTED = True

class BaseImage(object):
    """ Representation of an image by an array, an image-to-world affine matrix and a temporal spacing """
    volume = np.array([])
    affine = np.eye(4)
    dt = 1

    def WriteToNifti(self, filename):
        nim = nib.Nifti1Image(self.volume, self.affine)
        nim.header['pixdim'][4] = self.dt
        nim.header['sform_code'] = 1
        nib.save(nim, filename)
        
class MRIFunctions:
    def __init__(self):

        self.db = psycopg2.connect(host='staging.dyadmed.com', database='libby', user='admin', password='C5mvjumqqmp!')
        self.mycursor = self.db.cursor()
        self.username = 'dyadadmin'
        self.password = 'Dyadmed2021'
        
        session = boto3.Session(
                        aws_access_key_id='AKIAX6HXN2UPZYZVMD72',
                        aws_secret_access_key='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',
                    )

        self.s3 = session.client('s3')
        self.s3_resource = session.resource('s3')
        self.bucket_name = 'viewer.dyadmed.com'
        
        
        
    def get_all_instance_ids_and_source(self, mri_study_id):
        try:
            sql  = "SELECT id, type FROM image_source WHERE id IN \
                (SELECT image_source_id FROM instance_dicom WHERE instance_uid IN  \
                (SELECT DISTINCT instance_uid FROM instance_dicom WHERE series_uid IN \
                (SELECT DISTINCT series_uid FROM series_dicom WHERE study_uid=%s)))"

            self.mycursor.execute(sql, (str(mri_study_id),))

            data_list = self.mycursor.fetchall()
            return data_list
        except Exception as e:
            raise Exception
    
    def get_aws_dir_for_instance(self, image_source_id):
        try:
            sql = "SELECT source FROM image_source_alt WHERE image_source_id=%s"
            self.mycursor.execute(sql, (str(image_source_id),))
            aws_dir = self.mycursor.fetchone()[0]
            self.db.commit()
            return aws_dir
        except Exception as e:
            raise Exception
        
    def get_PACS_ID_for_instance(self, image_source_id):
        try:
            sql = "SELECT source FROM image_source WHERE id=%s"
            self.mycursor.execute(sql, (str(image_source_id),))
            PACS_dir = self.mycursor.fetchone()[0]
            self.db.commit()
            PACS_ID = PACS_dir.split('instances/')[1]
            return PACS_ID
        except Exception as e:
            # Logging the error
            raise Exception
        
    def create_sax_vol(self, list_pydcm):
        try:
            series_groups = {}
            series_z_pos = set()
            for f in list_pydcm:
                m = re.match('CINE_segmented_SAX_b(\d*)$',f.SeriesDescription)
                if m:
                    series_groups[f.SeriesDescription] = series_groups.get(f.SeriesDescription,[]) + [f]
                    tup = (f.SeriesDescription,int(m.group(1)))
                    if tup not in series_z_pos:
                        series_z_pos.add(tup)
            for k in series_groups:
                series_groups[k] = sorted(series_groups[k],key=lambda x:x.SeriesNumber)
            series_ordered_by_index = sorted(list(series_z_pos),key=lambda x:x[1])

            Z = len(series_ordered_by_index)
            d = series_groups[series_ordered_by_index[0][0]][0]

            T = d.CardiacNumberOfImages

            X = d.Columns
            Y = d.Rows
            T = d.CardiacNumberOfImages
            dx = float(d.PixelSpacing[1])
            dy = float(d.PixelSpacing[0])

        
            # The coordinate of the upper-left voxel of the first and second slices
            pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
            pos_ul[:2] = -pos_ul[:2]

            # Image orientation
            axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
            axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
            axis_x[:2] = -axis_x[:2]
            axis_y[:2] = -axis_y[:2]

            if Z >= 2:
                # Read a dicom file at the second slice
                d2 = series_groups[series_ordered_by_index[1][0]][0]
                pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
                pos_ul2[:2] = -pos_ul2[:2]
                axis_z = pos_ul2 - pos_ul
                axis_z = axis_z / np.linalg.norm(axis_z)
            else:
                axis_z = np.cross(axis_x, axis_y)

            # Determine the z spacing
            if hasattr(d, 'SpacingBetweenSlices'):
                dz = float(d.SpacingBetweenSlices)
            elif Z >= 2:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                    'Calculate from two successive slices.')
                dz = float(np.linalg.norm(pos_ul2 - pos_ul))
            else:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                    'Use attribute SliceThickness instead.')
                dz = float(d.SliceThickness)

            # Affine matrix which converts the voxel coordinate to world coordinate
            affine = np.eye(4)
            affine[:3, 0] = axis_x * dx
            affine[:3, 1] = axis_y * dy
            affine[:3, 2] = axis_z * dz
            affine[:3, 3] = pos_ul

            # The 4D volume
            volume = np.zeros((X, Y, Z, T), dtype='float32')
            

            name_to_position = {}
            position_to_name = {}

            for z in range(0, Z):
                # In a few cases, there are two or three time sequences or series within each folder.
                # We need to find which seires to convert.
                files = series_groups[series_ordered_by_index[z][0]]

                # Now for this series, sort the files according to the trigger time.
                files_time = []
                for f in files:
                    #d = dicom.read_file(os.path.join(dir[z], f))
                    d = f
                    t = d.TriggerTime
                    files_time += [[f, t]]
                files_time = sorted(files_time, key=lambda x: x[1])

                # Read the images
                for t in range(0, T):
                    # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
                    # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
                    # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
                    # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
                    # with nibabel's dimension.
                    try:
                        f = files_time[t][0]
                        #d = dicom.read_file(os.path.join(dir[z], f))
                        d = f
                        volume[:, :, z, t] = d.pixel_array.transpose()
                        name_to_position[f.SOPInstanceUID] = (z,t)
                        position_to_name[(z,t)] = f.SOPInstanceUID
                    except IndexError:
                        print('Warning: dicom file missing for {0}: time point {1}. '
                            'Image will be copied from the previous time point.'.format(dir[z], t))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                        position_to_name[(z,t)] = position_to_name[(z,t-1)]
                    except (ValueError, TypeError):
                        print('Warning: failed to read pixel_array from file {0}. '
                            'Image will be copied from the previous time point.'.format(os.path.join(dir[z], f)))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                        position_to_name[(z,t)] = position_to_name[(z,t-1)]
                    except NotImplementedError:
                        print('Warning: failed to read pixel_array from file {0}. '
                            'pydicom cannot handle compressed dicom files. '
                            'Switch to SimpleITK instead.'.format(os.path.join(dir[z], f)))
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(os.path.join(dir[z], f))
                        img = sitk.GetArrayFromImage(reader.Execute())
                        volume[:, :, z, t] = np.transpose(img[0], (1, 0))
                        position_to_name[(z,t)] = f.SOPInstanceUID


            # Temporal spacing
            dt = (files_time[1][1] - files_time[0][1]) * 1e-3

            # Store the image
            nifty_img = BaseImage()
            nifty_img.volume = volume
            nifty_img.affine = affine
            nifty_img.dt = dt

            return volume,position_to_name,name_to_position,nifty_img
        except Exception as e:
        # Logging the error
            raise Exception
        
    def create_lax_vol(self, list_pydcm):
        try:
            series_groups = {}
            series_z_pos = set()
            for f in list_pydcm:
                #m = re.match('CINE_segmented_SAX_b(\d*)$',f.SeriesDescription)
                m = re.match('CINE_segmented_LAX_4Ch$',f.SeriesDescription)
                if m:
                    series_groups[f.SeriesDescription] = series_groups.get(f.SeriesDescription,[]) + [f]
                    #series_groups[1] = series_groups.get(1,[])+[f]
                    #tup = (f.SeriesDescription,int(m.group(1)))
                    tup = (f.SeriesDescription,1)
                    if tup not in series_z_pos:
                        series_z_pos.add(tup)
                    #     series_z_pos = set([1])
            for k in series_groups:
                series_groups[k] = sorted(series_groups[k],key=lambda x:x.SeriesNumber)
            series_ordered_by_index = sorted(list(series_z_pos),key=lambda x:x[1])
            #
            Z = len(series_ordered_by_index)
            d = series_groups[series_ordered_by_index[0][0]][0]
            #
            T = d.CardiacNumberOfImages
        
            X = d.Columns
            Y = d.Rows
            T = d.CardiacNumberOfImages
            dx = float(d.PixelSpacing[1])
            dy = float(d.PixelSpacing[0])
            #
            # DICOM coordinate (LPS)
            #  x: left
            #  y: posterior
            #  z: superior
            # Nifti coordinate (RAS)
            #  x: right
            #  y: anterior
            #  z: superior
            # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
            # Refer to
            # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
            # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

            # The coordinate of the upper-left voxel of the first and second slices
            pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
            pos_ul[:2] = -pos_ul[:2]

            # Image orientation
            axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
            axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
            axis_x[:2] = -axis_x[:2]
            axis_y[:2] = -axis_y[:2]

            if Z >= 2:
                # Read a dicom file at the second slice
                d2 = series_groups[series_ordered_by_index[1][0]][0]
                pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
                pos_ul2[:2] = -pos_ul2[:2]
                axis_z = pos_ul2 - pos_ul
                axis_z = axis_z / np.linalg.norm(axis_z)
            else:
                axis_z = np.cross(axis_x, axis_y)

            # Determine the z spacing
            if hasattr(d, 'SpacingBetweenSlices'):
                dz = float(d.SpacingBetweenSlices)
            elif Z >= 2:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                    'Calculate from two successive slices.')
                dz = float(np.linalg.norm(pos_ul2 - pos_ul))
            else:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                    'Use attribute SliceThickness instead.')
                dz = float(d.SliceThickness)

            # Affine matrix which converts the voxel coordinate to world coordinate
            affine = np.eye(4)
            affine[:3, 0] = axis_x * dx
            affine[:3, 1] = axis_y * dy
            affine[:3, 2] = axis_z * dz
            affine[:3, 3] = pos_ul

            # The 4D volume
            volume = np.zeros((X, Y, Z, T), dtype='float32')
            name_to_position = {}
            position_to_name = {}

            for z in range(0, Z):
                # In a few cases, there are two or three time sequences or series within each folder.
                # We need to find which seires to convert.
                files = series_groups[series_ordered_by_index[z][0]]

                # Now for this series, sort the files according to the trigger time.
                files_time = []
                for f in files:
                    #d = dicom.read_file(os.path.join(dir[z], f))
                    d = f
                    t = d.TriggerTime
                    files_time += [[f, t]]
                files_time = sorted(files_time, key=lambda x: x[1])

                # Read the images
                for t in range(0, T):
                    # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
                    # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
                    # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
                    # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
                    # with nibabel's dimension.
                    try:
                        f = files_time[t][0]
                        #d = dicom.read_file(os.path.join(dir[z], f))
                        d = f
                        volume[:, :, z, t] = d.pixel_array.transpose()
                        name_to_position[f.SOPInstanceUID] = (z,t)
                        position_to_name[(z,t)] = f.SOPInstanceUID
                    except IndexError:
                        print('Warning: dicom file missing for {0}: time point {1}. '
                            'Image will be copied from the previous time point.'.format(dir[z], t))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                        position_to_name[(z,t)] = position_to_name[(z,t-1)]
                    except (ValueError, TypeError):
                        print('Warning: failed to read pixel_array from file {0}. '
                            'Image will be copied from the previous time point.'.format(os.path.join(dir[z], f)))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                        position_to_name[(z,t)] = position_to_name[(z,t-1)]
                    except NotImplementedError:
                        print('Warning: failed to read pixel_array from file {0}. '
                            'pydicom cannot handle compressed dicom files. '
                            'Switch to SimpleITK instead.'.format(os.path.join(dir[z], f)))
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(os.path.join(dir[z], f))
                        img = sitk.GetArrayFromImage(reader.Execute())
                        volume[:, :, z, t] = np.transpose(img[0], (1, 0))
                        position_to_name[(z,t)] = f.SOPInstanceUID

            # Temporal spacing
            dt = (files_time[1][1] - files_time[0][1]) * 1e-3

            # Store the image
            nifty_img = BaseImage()
            nifty_img.volume = volume
            nifty_img.affine = affine
            nifty_img.dt = dt

            return volume,position_to_name,name_to_position,nifty_img
        except Exception as e:
            raise Exception
    def _get_padding(self, a,b):
        try:
            diff = b-a
            assert diff>0
            if diff % 2 == 0:
                return int(diff/2),int(diff/2)
            else:
                return int((diff+1)/2-1),int((diff+1)/2)
        except Exception as e:
            raise Exception
    
    def predict_vol_pytorch_by_crops(self, vol,inferer_segmentation,model_input_shape,num_classes=4):
        try:
            print("vol orig shape",vol.shape)
            
            predicted = []
            model_device = str(list(inferer_segmentation.parameters())[0].device)
            for i in range(vol.shape[0]):
                #resized = skimage.transform.resize(vol[i],(model.input_shape[1],model.input_shape[2]))[np.newaxis,np.newaxis,:,:].astype(np.float32)
                #resized = torch.tensor( resized ).to(model_device)
                
                x = vol[i].copy()
                #print("x orig shape",x.shape)
                h,w = (x.shape[0],x.shape[1])
                #print("h w",h,w)
                pad_up,pad_down,pad_left,pad_right = 0,0,0,0
                if x.shape[0]<model_input_shape[1]:
                    pad_up,pad_down = self._get_padding(x.shape[0],model_input_shape[1])
                if x.shape[1]<model_input_shape[2]:
                    pad_left,pad_right = self._get_padding(x.shape[1],model_input_shape[2])
                #print("pad up down left right",pad_up,pad_down,pad_left,pad_right)
                #x = np.pad(x,((pad_up,pad_down),(pad_left,pad_right)),
                #                        'constant',constant_values=0)
                x = np.pad(x,((pad_up,pad_down),(pad_left,pad_right)),
                                        'minimum')
                #print("x shape",x.shape)
                first_dim_starts = [0]
                if model_input_shape[1] > x.shape[0]:
                    next_start = model_input_shape[1]
                    while next_start+model_input_shape[1]<=x.shape[1]:
                        first_dim_starts.append(next_start)
                        next_start += model_input_shape[1]
                    first_dim_starts.append(x.shape[0]-model_input_shape[1])
                second_dim_starts = [0]                    
                if model_input_shape[2] > x.shape[1]:
                    next_start = model_input_shape[2]
                    while next_start+model_input_shape[2]<=x.shape[2]:
                        second_dim_starts.append(next_start)
                        next_start += model_input_shape[2]
                    second_dim_starts.append(vol.shape[2]-model_input_shape[2])
                    
                crops = []
                crop_starts = []
                for s1 in first_dim_starts:
                    for s2 in second_dim_starts:
                        crops.append(x[s1:s1+model_input_shape[1],s2:s2+model_input_shape[2]][np.newaxis,:,:])
                        crop_starts.append((s1,s2))
                print("crops: " + str(crops[0].shape))
                preds_reoriented = []
                for j in range(len(crops)):
                    #current_pred = model(torch.tensor(crops[j].astype(np.float32)).to(model_device)).detach().cpu().numpy()[0]
                    current_pred = inferer_segmentation(torch.tensor(np.expand_dims(crops[j], axis = 0).astype(np.float32)).to(model_device)).detach().cpu().numpy()[0]
                    pred_reoriented = np.full([num_classes,x.shape[0], x.shape[1]], np.nan)
                    w1,w2 = crop_starts[j][0],crop_starts[j][0]+model_input_shape[1]
                    w3,w4 = crop_starts[j][1],crop_starts[j][1]+model_input_shape[2]
                    pred_reoriented[:,w1:w2,w3:w4] = current_pred
                    preds_reoriented.append(pred_reoriented)
                    #print(pred_reoriented.shape)
                
                preds_averaged = np.nanmean(np.array(preds_reoriented),axis=0)
                predicted.append(preds_averaged[:,pad_up:pad_up+h, pad_left:pad_left+w].argmax(0))
                #print("last predicted shape",predicted[-1].shape)
            return np.array(predicted)
        except Exception as e:
            print(e)
            raise Exception
    def remove_small_connected_components(self, mask,num_classes=6):
        try:
            out_img = np.zeros(mask.shape, dtype=np.uint8)

            for struc_id in range(1,num_classes):

                binary_img = mask == struc_id
                blobs = skimage.measure.label(binary_img, connectivity=1)

                props = skimage.measure.regionprops(blobs)

                if not props:
                    continue

                area = [ele.area for ele in props]
                num_slices = [(i,ele.slice[0].stop-ele.slice[0].start) for i,ele in enumerate(props)]
                sort_by_area_and_height = sorted(num_slices,key=lambda x:-x[1])
                largest_blob_ind = np.argmax(area)
                largest_blob_label = props[largest_blob_ind].label

                out_img[blobs == largest_blob_label] = struc_id

            return out_img
        except Exception as e:
            raise Exception
    def remove_small_connected_components_select_tallest(self, mask):
        try:
            out_img = np.zeros(mask.shape, dtype=np.uint8)
            for struc_id in [1, 2, 3]:

                binary_img = mask == struc_id
                blobs = skimage.measure.label(binary_img, connectivity=1)

                props = skimage.measure.regionprops(blobs)

                if not props:
                    continue

                area = [ele.area for ele in props]
                num_slices = [(i,ele.slice[0].stop-ele.slice[0].start) for i,ele in enumerate(props)]
                sort_by_area_and_height = sorted(num_slices,key=lambda x:-x[1])
                #largest_blob_ind = np.argmax(area)
                largest_blob_ind = sort_by_area_and_height[0][0]
                #if largest height not more than 1 away than 2nd largest take largest area
                largest_blob_label = props[largest_blob_ind].label

                out_img[blobs == largest_blob_label] = struc_id

            return out_img
        except Exception as e:
            raise Exception
    
    def post_proc_mask_select_comp(self, mask):
        try:
            out_img = self.remove_small_connected_components_select_tallest(mask)
            lv_filled = scipy.ndimage.morphology.binary_fill_holes((out_img==3)*1)
            rv_filled = scipy.ndimage.morphology.binary_fill_holes((out_img==1)*1)
            myo_filled = scipy.ndimage.morphology.binary_fill_holes((out_img==2)*1)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    for k in range(mask.shape[2]):
                        if lv_filled[i,j,k] == 1:
                            out_img[i,j,k] = 3
                        elif rv_filled[i,j,k] == 1:
                            out_img[i,j,k] = 1
                        elif myo_filled[i,j,k] == 1 and out_img[i,j,k] == 0:
                            out_img[i,j,k] = 2
            print("post_proc_mask shape {}".format(out_img.shape))
            return out_img
        except Exception as e:
            raise Exception
    
    def test_write_db(self, dcm_img_list,jf=None):
        if jf is None:
            jf = json.load(open("/app/mri_inference_engine/20210402-231359-477057-json_output_folder/1.2.826.0.1.4978912.2202.20190416180353.json"))
            jf = json.load(open("/app/mri_inference_engine/20210402-200155-230894-json_output_folder/1.2.826.0.1.4978912.2202.20190607154540.json"))
            jf = json.load(open("/app/mri_inference_engine/20210412-224812-656501-json_output_folder/1.2.826.0.1.497891220150205131440.json"))
        """series_info_dict = {}
        study_uid = None
        for f in sax_files:
            m = re.match('CINE_segmented_SAX_b(\d*)$',f.SeriesDescription)
            if m:
                if f.SeriesInstanceUID not in series_info_dict:
                    info_dict = {}
                    info_dict["series_number"] = f.SeriesNumber
                    info_dict["series_date"] = f.SeriesDate
                    info_dict["series_time"] = f.SeriesTime
                    series_info_dict[f.SeriesInstanceUID] = info_dict
            study_uid = f.StudyInstanceUID"""

        instance_result = {}
        for series in jf["Series"]:
            for instance in jf["Series"][series]["Instances"]:
                instance_result[instance] = jf["Series"][series]["Instances"][instance]
        self.update_patient_dicom_table(dcm_img_list)
        self.update_study_dicom_table(dcm_img_list,jf)
        self.update_series_dicom_table(dcm_img_list,jf)
        for f in dcm_img_list:
            #connection.update_image_source_table(f)
            inst_res = None
            try:
                inst_res = instance_result[f.SOPInstanceUID]
            except Exception as e:
                print("exception in getting inst_res",e)

            self.update_instance_table_from_Dicom(f,inst_res)
    def update_patient_dicom_table(self, sax_files):
        try:
            timestamp = datetime.fromtimestamp(int(time.time()))
            try:
                patient_uid = sax_files[0].PatientID
            except Exception as e:
                patient_uid = sax_files[0].StudyInstanceUID
            sql = "INSERT INTO patient_dicom (patient_uid,timestamp) VALUES (%s,%s)"    

            try:
                self.mycursor.execute(sql, (patient_uid, str(timestamp)))
            except Exception as e:
                print("Database connection failed due to {}".format(e))
                try:
                    self.db.rollback()
                    sql = "UPDATE patient_dicom SET modified=%s WHERE patient_uid=%s"
                    self.mycursor.execute(sql, (str(timestamp),str(patient_uid)))
                except Exception as e:
                    print("Database connection failed due to {}".format(e))
                    raise Exception

            self.db.commit()

        except Exception as e:
            raise Exception
    def update_study_dicom_table(self, sax_files,json_dict=None,json_file=None):
        try:
            timestamp = datetime.fromtimestamp(int(time.time()))
            if json_dict is not None:
                output_json = json_dict
            else:
                output_json = json.load(open(json_file))
            sample_series_name = list(output_json["Series"].keys())[0]
            sample_instance_name = list(output_json["Series"][sample_series_name]["Instances"])[0]
            sample_instance_dict = output_json["Series"][sample_series_name]["Instances"][sample_instance_name]
            study_uid = str(sample_instance_dict['Instance General Information']['StudyInstanceUID'])
            study_date = str(int(output_json["Meta Information"]["StudyDate"]))
            #study_time = str(int(output_json["Meta Information"]["StudyTime"]))
            study_time = str(int(float(str(output_json["Meta Information"]["StudyTime"]))))

            try:
                manufacturer = str(sax_files[0].Manufacturer)
            except Exception as e:
                manufacturer = None
            try:
                manufacturer_model_name = str(sax_files[0].ManufacturerModelName)
            except Exception as e:
                manufacturer_model_name = None
            try:
                institution_name = str(sax_files[0].InstitutionName)
            except Exception as e:
                institution_name = None

            self.mycursor.execute("SELECT * FROM study_dicom WHERE study_uid=%s",(study_uid,))
            json_str = None
            if output_json is not None:
                output_json_copy = output_json.copy()
                #del output_json_copy["Series"]
                output_json_copy["Series"] = list(output_json_copy["Series"].keys())
                json_str = str(json.dumps(output_json_copy))
            
            if self.mycursor.fetchone() is not None:
                print("study_uid",str(study_uid),"already in db")
                try:
                    
                    sql = "UPDATE study_dicom SET modified=%s, study_date=%s, study_time=%s, modality=%s, study_result=%s, json_result=%s WHERE study_uid=%s"
                    self.db.cursor().execute(sql, (str(timestamp),str(study_date),str(study_time),"MR","1",json_str,str(study_uid)))
                except Exception as e:
                    print("Database connection failed due to {}".format(e))
                    raise Exception
            else:
                print("study_uid",str(study_uid),"not in db")
                try:
                    self.db.rollback()
                    sql = "INSERT INTO study_dicom (study_uid, timestamp, patient_uid, study_date, study_time, modality, manufacturer, manufacturer_model_name, institution_name, study_result, json_result) VALUES (%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s)"    
                    self.db.cursor().execute(sql, (study_uid, str(timestamp), str(study_uid), study_date, study_time, "MR", manufacturer, manufacturer_model_name, institution_name, "1", json_str))
                except Exception as e:
                    print("Database connection failed due to {}".format(e))
                    raise Exception

            self.db.commit()
        except Exception as e:
            raise Exception    
    
    def update_series_dicom_table(self, sax_files,json_dict=None,json_file=None):
        try:
            timestamp = datetime.fromtimestamp(int(time.time()))
            if json_dict is not None:
                output_json = json_dict
            else:
                output_json = json.load(open(json_file))
            sql = "INSERT INTO series_dicom (series_uid, timestamp, study_uid, series_number, series_date, series_time, series_result, json_result) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"    

            series_info_dict = {}
            study_uid = None
            for f in sax_files:
                #m = re.match('CINE_segmented_SAX_b(\d*)$',f.SeriesDescription)
                #if m:
                if f.SeriesInstanceUID not in series_info_dict:
                    info_dict = {}
                    try:
                        info_dict["series_number"] = f.SeriesNumber
                    except:
                        info_dict["series_number"] = -1
                    try:
                        info_dict["series_date"] = f.SeriesDate
                    except:
                        info_dict["series_date"] = -1
                    try:
                        info_dict["series_time"] = f.SeriesTime
                    except:
                        info_dict["series_time"] = -1
                    series_info_dict[f.SeriesInstanceUID] = info_dict
                study_uid = f.StudyInstanceUID

            for series_uid in series_info_dict:
                try:
                    output_json_copy = output_json["Series"][series_uid].copy()
                    #del output_json_copy["Instances"]
                    output_json_copy["Instances"] = list(output_json_copy["Instances"].keys())
                    json_series = str(json.dumps(output_json_copy))
                    #json_series =  str(json.dumps(output_json["Series"][series_uid]))
                except Exception as e:
                    json_series = None

                try:
                    print("trying insert series_uid",series_uid)
                    info_dict = series_info_dict[series_uid]
                    self.db.cursor().execute(sql, (str(series_uid), str(timestamp), str(study_uid), str(info_dict["series_number"]), str(info_dict["series_date"]), str(info_dict["series_time"]), "1",json_series) )
                except Exception as e:
                    print("Database connection failed due to {}".format(e))
                    try:
                        self.db.rollback()
                        sql = "UPDATE series_dicom SET modified=%s, series_result=%s, json_result=%s WHERE series_uid=%s"
                        self.db.cursor().execute(sql, (str(timestamp),"1",json_series,str(series_uid)))
                    except Exception as e:
                        print("Database connection failed due to {}".format(e))
                        raise Exception

                self.db.commit()
        except Exception as e:
            raise Exception
    
    def update_instance_table_from_Dicom(self, dataset,json_result=None):
        try:
            print("updating instance")
            timestamp = datetime.fromtimestamp(int(time.time()))

            try:
                SOP_Instance_UID = dataset.SOPInstanceUID
                _, image_source_id = self.get_source_dir_for_instance(SOP_Instance_UID) # This function returns image source id along with source

            except:
                SOP_Instance_UID = None   
            else:   
                try:
                    Series_Instance_UID = dataset.SeriesInstanceUID
                except:
                    Series_Instance_UID = None
                try:
                    Instance_Number = dataset.InstanceNumber
                except:
                    Instance_Number = 0
                try:
                    Acquisition_Time = dataset.AcquisitionDateTime[0:14]
                    if Acquisition_Time == '':
                        Acquisition_Time = 0
                except:
                    Acquisition_Time = 0
                try:
                    Rows = dataset.Rows
                    if Rows == '':
                        Rows = 0
                except:
                    Rows = 0
                try:
                    Columns = dataset.Columns
                    if Columns == '':
                        Columns = 0
                except:
                    Columns = 0
                try:
                    Instance_Creation_Date = dataset.InstanceCreationDate
                    if Instance_Creation_Date == '':
                        Instance_Creation_Date = 0
                except:
                    Instance_Creation_Date = 0
                try:
                    Instance_Creation_Time = int(dataset.InstanceCreationTime)
                    if Instance_Creation_Time == '':
                        Instance_Creation_Time = 0
                except:
                    Instance_Creation_Time = 0
                    
                try:
                    Frame_Time = float(dataset.FrameTime)
                except:
                    Frame_Time = None
                    Frame_Rate = 0
                else:
                    Frame_Rate = int(round(1000 / Frame_Time))

                
                sql_exist = "select exists(select 1 from instance_dicom where instance_uid=%s)"
                self.mycursor.execute(sql_exist, (str(SOP_Instance_UID),))
                if self.mycursor.fetchone()[0] == False:
                
                    sql = "INSERT INTO instance_dicom (instance_uid, timestamp, image_source_id, series_uid, instance_number, instance_creation_date, instance_creation_time, \
                        acquisition_date_time, frame_rate, instance_width, instance_height, instance_depth,json_result) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"    
                    
                    try:    
                        self.mycursor.execute(sql, (str(SOP_Instance_UID), str(timestamp), str(image_source_id), str(Series_Instance_UID), str(Instance_Number), 
                                                    str(Instance_Creation_Date), str(Instance_Creation_Time), str(Acquisition_Time), str(Frame_Rate), 
                                                    str(Columns), str(Rows), None, str(json.dumps(json_result)) if json_result is not None else None )) 
                    except Exception as e:
                        print("Insert instance Database connection failed due to {}".format(e))
                        raise Exception
                else:
                    sql = "UPDATE instance_dicom SET modified=%s, frame_rate=%s, json_result=%s WHERE instance_uid=%s"
                    try:
                        self.mycursor.execute(sql, (str(timestamp), str(Frame_Rate), str(json.dumps(json_result)) if json_result is not None else None, str(SOP_Instance_UID)))
                    except Exception as e:
                        print("Update instance Database connection failed due to {}".format(e))
                        raise Exception
                self.db.commit()

        except Exception as e:
            raise Exception
    
    
    def create_json(self, list_pydcm, inferer_segmentation, inferer_segmentation_LAX):
        try:
            pydcm_by_SOPInstanceUID = {}
            for f in list_pydcm:
                pydcm_by_SOPInstanceUID[f.SOPInstanceUID] = f
            sa_4d_vol, position_to_name, name_to_position, nifty_img = self.create_sax_vol(list_pydcm)

            start_time_datetime = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            time_milestones = [timeit.default_timer()]
            patient_folder = start_time_datetime 
            os.makedirs(patient_folder)
            #sa_name = start_time_datetime+"-"+"sa.nii.gz"
            seg_sa_name = start_time_datetime+"-"+"seg_sa.nii.gz"
            #nifty_img.WriteToNifti(sa_name)
            nifty_img.WriteToNifti(os.path.join(patient_folder,"sa.nii.gz"))
            pred_3d_vols = []
            for i in range(sa_4d_vol.shape[3]):
                print("pred time",i)
                #time_pt_pred = MyUtilFunctions.predict_vol_pytorch_by_crops(sa_4d_vol[:,:,:,i].transpose([2,1,0]),unet_model)
                #time_pt_pred_postproc =  MyUtilFunctions.post_proc_mask_select_comp(time_pt_pred).transpose([2,1,0])
                time_pt_pred = self.predict_vol_pytorch_by_crops(sa_4d_vol[:,:,:,i].transpose([2,1,0]),inferer_segmentation,(1,256,256))
                time_pt_pred_postproc =  self.post_proc_mask_select_comp(time_pt_pred).transpose([2,1,0])

                pred_3d_vols.append(time_pt_pred_postproc)
            sa_4d_vol_pred = np.array(pred_3d_vols).transpose([1,2,3,0])
            #sa_4d_vol_pred = nib.load("20201014-212350-531807-sa.nii.gz").get_fdata()
            assert sa_4d_vol_pred.shape==sa_4d_vol.shape
            time_milestones.append(timeit.default_timer())
            pred_nifty = BaseImage()
            pred_nifty.volume = sa_4d_vol_pred
            pred_nifty.affine = nifty_img.affine
            pred_nifty.dt = nifty_img.dt
            pred_nifty.WriteToNifti(seg_sa_name)

            nim = nib.load(os.path.join(patient_folder,"sa.nii.gz"))
            pixdim = nim.header['pixdim'][1:4]
            volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3

            LV_vols = [Counter(sa_4d_vol_pred[:,:,:,t].flatten())[3]*volume_per_pix for t in range(sa_4d_vol_pred.shape[3])]
            Myo_vols = [Counter(sa_4d_vol_pred[:,:,:,t].flatten())[2]*volume_per_pix for t in range(sa_4d_vol_pred.shape[3])]
            RV_vols = [Counter(sa_4d_vol_pred[:,:,:,t].flatten())[1]*volume_per_pix for t in range(sa_4d_vol_pred.shape[3])]
            
            ES_frame = int(np.argmin(LV_vols))

            repeated_indices = {
                'Volumes': {'ED':{'LV':LV_vols[0],'RV':LV_vols[0],'MC':Myo_vols[0]},
                        'ES':{'LV':LV_vols[ES_frame],'RV':LV_vols[ES_frame],'MC':Myo_vols[ES_frame]}},
                'Ejection_Fraction':{"EF_list_from_volume":[100*(LV_vols[0]-LV_vols[ES_frame])/LV_vols[0]],
                                    "ED_list":[0],
                                    "ES_list":[ES_frame]},
                'Stroke_Volume':LV_vols[0]-LV_vols[ES_frame],
                'Masses': {'ED':{'LV':LV_vols[0]*1.05,'RV':LV_vols[0]*1.05,'MC':Myo_vols[0]*1.05},
                        'ES':{'LV':LV_vols[ES_frame]*1.05,'RV':LV_vols[ES_frame]*1.05,'MC':Myo_vols[ES_frame]*1.05}}, #1.05 g/ml
                'Segmentation': {'LV':16,'RV':32,'MC':64},
            }
            time_milestones.append(timeit.default_timer())

            sa_4d_vol_pred_relab_for_strain = sa_4d_vol_pred.copy()
            sa_4d_vol_pred_relab_for_strain[sa_4d_vol_pred==3] = 1
            sa_4d_vol_pred_relab_for_strain[sa_4d_vol_pred==1] = 3

            sa_nii_file = nifty_img
            new_seg_file = nib.Nifti1Image(sa_4d_vol_pred_relab_for_strain, sa_nii_file.affine)
            new_seg_file.header['pixdim'][4] = 1
            new_seg_file.header['sform_code'] = 1
            nib.save(new_seg_file, start_time_datetime+"-"+"seg_sa.nii.gz")
            nib.save(new_seg_file, os.path.join(patient_folder,"seg_sa.nii.gz"))
            print("wrote to ",start_time_datetime+"-"+"seg_sa.nii.gz")
            
            new_seg_file = nib.Nifti1Image(sa_4d_vol_pred_relab_for_strain[:,:,:,0], sa_nii_file.affine)
            #new_seg_file = nib.Nifti1Image(sa_4d_vol_gt[:,:,:,0], sa_nii_file.affine)
            new_seg_file.header['pixdim'][4] = 1
            new_seg_file.header['sform_code'] = 1
            nib.save(new_seg_file, start_time_datetime+"-"+"seg_sa_ED.nii.gz")
            nib.save(new_seg_file, os.path.join(patient_folder,"seg_sa_ED.nii.gz"))
            print("wrote to ",start_time_datetime+"-"+"seg_sa_ED.nii.gz")

            new_seg_file = nib.Nifti1Image(sa_4d_vol_pred_relab_for_strain[:,:,:,ES_frame], sa_nii_file.affine)
            #new_seg_file = nib.Nifti1Image(sa_4d_vol_gt[:,:,:,ES_frame], sa_nii_file.affine)
            new_seg_file.header['pixdim'][4] = 1
            new_seg_file.header['sform_code'] = 1
            nib.save(new_seg_file, start_time_datetime+"-"+"seg_sa_ES.nii.gz")
            nib.save(new_seg_file, os.path.join(patient_folder,"seg_sa_ES.nii.gz"))
            print("wrote to ",start_time_datetime+"-"+"seg_sa_ES.nii.gz")

            #add lax analysis
            self.create_lax_segmentations(list_pydcm,start_time_datetime, inferer_segmentation_LAX)
            time_milestones.append(timeit.default_timer())
            gls_result = None
            for data_dir in [patient_folder]:
                seg_la_name = '{0}/seg4_la_4ch_ED.nii.gz'.format(data_dir)
                if not os.path.exists(seg_la_name):
                    gls_result = "segmentation does not exist"
                    continue
                if not card_utils.la_pass_quality_control(seg_la_name):
                    gls_result = "segmentation did not pass quality control"
                    continue

                # Intermediate result directory
                motion_dir = os.path.join(data_dir, 'cine_motion')
                if not os.path.exists(motion_dir):
                    os.makedirs(motion_dir)

                # Perform motion tracking on long-axis images and calculate the strain
                try:
                    strain_table = card_utils.cine_2d_la_motion_and_strain_analysis(data_dir,
                                                    "ukbb_cardiac/par",
                                                    motion_dir,
                                                    '{0}/strain_la_4ch'.format(data_dir))
                    gls_result = [strain_table[i,:].min() for i in range(7)]
                except Exception as e:
                    gls_result = str(e)
                #repeated_indices["gls"] = strain_table.tolist()        
                
            repeated_indices["gls"] = gls_result
            time_milestones.append(timeit.default_timer())

            #patient_folder = start_time_datetime 
            #os.makedirs(patient_folder)
            labels = {'LV': 1, 'Myo': 2, 'RV': 3}
            seg_sa_name = '{0}/seg_sa_ED.nii.gz'.format(patient_folder)
            #import shutil
            #shutil.copy2(start_time_datetime+"-"+"seg_sa_ED.nii.gz",seg_sa_name)
            calc_strain = False
            if not os.path.exists(seg_sa_name):
                print("seg_sa dne")
                calc_strain = False
            if not card_utils.sa_pass_quality_control(seg_sa_name,label=labels):
                print("did not pass qc")
                calc_strain = False

            strain_array = None
            if calc_strain:
                motion_dir = os.path.join(patient_folder, 'cine_motion')
                if not os.path.exists(motion_dir):
                    os.makedirs(motion_dir)
                    print("made ",motion_dir)
                print(motion_dir)
                print(os.path.abspath(motion_dir))
                labels = {'LV': 1, 'Myo': 2, 'RV': 3} #wenjaibai convention
                strain_array = card_utils.cine_2d_sa_motion_and_strain_analysis(os.path.abspath(patient_folder),
                                                            os.path.abspath("ukbb_cardiac/par"),
                                                            os.path.abspath(motion_dir),
                                                            '{0}/strain_sa'.format(patient_folder),label=labels)
            
                strain_dict_radial = {"S"+str(i+1):strain_array['radial'][i,:].max() for i in range(17)}
                strain_dict_circum = {"S"+str(i+1):strain_array['circum'][i,:].min() for i in range(17)}
            time_milestones.append(timeit.default_timer())

            if CARDUTILS_IMPORTED:
                try:
                    thickness_table = card_utils.evaluate_wall_thickness(start_time_datetime+"-"+'seg_sa_ED.nii.gz',
                                                            start_time_datetime+"-"+'wall_thickness_ED',label=labels)
                    repeated_indices['Myocardial_Thickness_ED'] =  {"S"+str(i+1):float(thickness_table[i]) for i in range(17)}
                except Exception as e:
                    print("error in thickness calculation",e)
                    repeated_indices['Myocardial_Thickness_ED'] = 'error '+str(e)
                if strain_array:
                    repeated_indices['Strain'] = {'RS':strain_dict_radial.copy(),"CS":strain_dict_circum.copy()}
            time_milestones.append(timeit.default_timer())
            
            sa_4d_vol_pred_relabeled = sa_4d_vol_pred.copy()
            sa_4d_vol_pred_relabeled[sa_4d_vol_pred==1] = repeated_indices['Segmentation']['RV']
            sa_4d_vol_pred_relabeled[sa_4d_vol_pred==3] = repeated_indices['Segmentation']['LV']
            sa_4d_vol_pred_relabeled[sa_4d_vol_pred==2] = repeated_indices['Segmentation']['MC']
            
            #zip_file_name = glob.glob(patient_folder+"/*20209_2_0.zip")[0]
            '''zip_file_name = "/home/john/Data/MRI/UKBiobank/short-axis/{}_20209_2_0.zip".format(eid)
            zf = zipfile.ZipFile(zip_file_name)
            import ukbb_cardiac.data.biobank_utils_edited as ukb_data_utils
            bb_obj = ukb_data_utils.Biobank_Dataset(os.path.join(patient_folder,"dicom"))
            bb_obj.read_dicom_images()'''

            output_json = {"Study Level Results":repeated_indices.copy()}
            series_json = {}
            series_ids_count = Counter()
            json_output_folder = start_time_datetime+"-"+"json_output_folder"
            os.makedirs(json_output_folder)
            s3_c= boto3.client('s3')
            for z in range(sa_4d_vol.shape[2]):
                slice_folder = os.path.join(json_output_folder,str(z))
                os.makedirs(slice_folder)
                #series_json["Series_%d"%z] = {"Instances":{},"Series_General_Information":{},"Series_Results":{}}
                for t in range(sa_4d_vol.shape[3]):
                    #pydcm_file = pydicom.dcmread( io.BytesIO(zf.open(bb_obj.data['sa'].position_to_name[(z,t)]).read()) )
                    pydcm_file = pydcm_by_SOPInstanceUID[position_to_name[(z,t)]]
                    skimage.io.imsave(os.path.join(slice_folder,"%0.4d.png"%t),sa_4d_vol[:,:,z,t].transpose())
                    skimage.io.imsave(os.path.join(slice_folder,"%0.4d_mask.png"%t),sa_4d_vol_pred_relabeled[:,:,z,t].transpose())
                    s3_c.upload_file(os.path.join(slice_folder,"%0.4d.png"%t),self.bucket_name,"public/HospitalA/MRI_results_repo_local_test/{}/{}/{}/{}.png".format(pydcm_file.StudyInstanceUID,pydcm_file.SeriesInstanceUID,pydcm_file.SOPInstanceUID,t))
                    s3_c.upload_file(os.path.join(slice_folder,"%0.4d_mask.png"%t),self.bucket_name,"public/HospitalA/MRI_results_repo_local_test/{}/{}/{}/{}_mask.png".format(pydcm_file.StudyInstanceUID,pydcm_file.SeriesInstanceUID,pydcm_file.SOPInstanceUID,t))
                    instance_json = {}

                    '''output_json['General Information'] = {'ImageWidth':int(sa_4d_vol.shape[0]),
                                                            'ImageHeight':int(sa_4d_vol.shape[1]),
                                                    'Date': str(pydcm_file.AcquisitionDate),
                                                        'Date': str(pydcm_file.AcquisitionTime),
                                                        'Gender': str(pydcm_file.PatientSex),
                                                        'Age': str(pydcm_file.PatientAge),
                                                        'Modality':'CardiacMRI'}'''
                    '''instance_json['Meta Information'] = {'PatientID':str(pydcm_file.PatientID),
                                                        'StudyID':str(pydcm_file.StudyID),
                                                        'StudyInstanceUID':str(pydcm_file.StudyInstanceUID),
                                                        'SeriesNumber':str(pydcm_file.SeriesNumber),
                                                        'SeriesInstanceUID':str(pydcm_file.SeriesInstanceUID),
                                                        'InstanceNumber':str(pydcm_file.InstanceNumber)}'''
                    '''output_json['Meta Information'] = {'Modality':'CardiacMRI',
                                                    'AnalysisDate':None ,
                                                    'PatientID':str(pydcm_file.PatientID),
                                                        'AnalysisTime': None,
                                                    'Gender': str(pydcm_file.PatientSex),
                                                    'PatientName':str(pydcm_file.PatientName),
                                                    'PatientBirthDate':str(pydcm_file.PatientBirthDate),
                                                        'Age': str(pydcm_file.PatientAge),
                                                        'StudyDate':str(pydcm_file.AcquisitionDate),
                                                    'StudyTime':str(pydcm_file.AcquisitionTime)
                                                    } #includes private tags'''
                    output_json['Meta Information'] = {'Modality':'CardiacMRI',
                                                    'AnalysisDate':None ,
                                                        'AnalysisTime': None,
                                                        'StudyDate':str(pydcm_file.AcquisitionDate),
                                                    'StudyTime':str(pydcm_file.AcquisitionTime)
                                                    }
                    series_id = pydcm_file.SeriesInstanceUID
                    series_ids_count[series_id] += 1
                    study_id = pydcm_file.StudyInstanceUID
                    instance_id = pydcm_file.SOPInstanceUID
                    if series_id not in series_json:
                        series_json[series_id] =  {"Instances":{},"Series Results":{},"Series General Information":{}}
                        series_json[series_id]['Series General Information'] = {'ImageWidth':int(sa_4d_vol.shape[0]),
                                                            'ImageHeight':int(sa_4d_vol.shape[1]),
                                                    'SourceType':None

                                                        }
                    #series_json["Series_%d"%z]["Instances"]["Instance_%d"%t] = {"Instance_Result":{}}
                    series_json[series_id]["Instances"][instance_id] = {"Instance Result":{}}
                    #series_json["Series_%d"%z]["Instances"]["Instance_%d"%t]['Instance_General_Information'] = {'PatientID':str(pydcm_file.PatientID),
                    series_json[series_id]["Instances"][instance_id]['Instance General Information'] = {'StudyID':str(pydcm_file.StudyID),
                                                        'StudyInstanceUID':str(pydcm_file.StudyInstanceUID),
                                                        'SeriesNumber':str(pydcm_file.SeriesNumber),
                                                        'SeriesInstanceUID':str(pydcm_file.SeriesInstanceUID),
                                                        'InstanceNumber':str(pydcm_file.InstanceNumber)}
                    #series_json["Series_%d"%z]["Instances"]["Instance_%d"%t]['Instance_Result']["Frame1"] = {"Segmentation":os.path.join(str(z),"%0.4d_mask.png"%t)}
                    series_json[series_id]["Instances"][instance_id]['Instance Result']["Frame1"] = {"Segmentation":os.path.join(str(z),"%0.4d_mask.png"%t)}
            output_json["Series"] = series_json
                    
            json.dump(output_json,open(os.path.join(json_output_folder,study_id+".json"),"w"))
            s3_c.upload_file(os.path.join(json_output_folder,study_id+".json"),self.bucket_name,"public/HospitalA/MRI_results_repo_local_test/{}/{}.json".format(pydcm_file.StudyInstanceUID,study_id))    
            time_milestones.append(timeit.default_timer())
            
            self.test_write_db(list_pydcm,output_json)
            
            time_milestones.append(timeit.default_timer())
            print(time_milestones)
            print([time_milestones[i]-time_milestones[i-1] for i in range(1,len(time_milestones))])
            return output_json, start_time_datetime
        except Exception as e:
            print(e)
            raise Exception

    def create_lax_segmentations(self,list_pydcm,start_time_datetime, inferer_segmentation_LAX):
        pydcm_by_SOPInstanceUID = {}
        for f in list_pydcm:
            pydcm_by_SOPInstanceUID[f.SOPInstanceUID] = f
        la_4d_vol, position_to_name_la, name_to_position_la, nifty_img_la = self.create_lax_vol(list_pydcm)

        #start_time_datetime = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        time_milestones = [timeit.default_timer()]
        patient_folder = start_time_datetime 
        #os.makedirs(patient_folder)

        la_name = start_time_datetime+"-"+"la_4ch.nii.gz"
        seg_la_name = start_time_datetime+"-"+"seg_la_4ch.nii.gz"
        #nifty_img.WriteToNifti(la_name)
        nifty_img_la.WriteToNifti(os.path.join(patient_folder,"la_4ch.nii.gz"))
        pred_lax_vols = []
        for i in range(la_4d_vol.shape[3]):
            print("pred time",i)
            time_pt_pred = self.predict_vol_pytorch_by_crops(la_4d_vol[:,:,:,i].transpose([2,1,0]),inferer_segmentation_LAX,(1,256,256),num_classes=6)
            #time_pt_pred_postproc =  post_proc_mask_select_comp(time_pt_pred).transpose([2,1,0])
            time_pt_pred_postproc = time_pt_pred.transpose([2,1,0])
            pred_lax_vols.append(time_pt_pred_postproc)
        la_4d_vol_pred = np.array(pred_lax_vols).transpose([1,2,3,0])
        def relabel_volume(vol,mapping):
            new_vol = vol.copy()
            for x in mapping:
                new_vol[vol==x] = mapping[x]
            return new_vol
        la_4d_vol_pred = relabel_volume(la_4d_vol_pred,{0:0,1:3,2:2,3:1,4:5,5:4}).astype("<f8")
        assert la_4d_vol_pred.shape==la_4d_vol.shape
        time_milestones.append(timeit.default_timer())
        pred_nifty = BaseImage()
        pred_nifty.volume = la_4d_vol_pred
        pred_nifty.affine = nifty_img_la.affine
        pred_nifty.dt = nifty_img_la.dt
        #pred_nifty.WriteToNifti(seg_la_name)
        pred_nifty.WriteToNifti(os.path.join(patient_folder,"seg4_la_4ch.nii.gz"))

        ED_seg_file = nib.Nifti1Image(la_4d_vol_pred[:,:,:,0], pred_nifty.affine)
        ED_seg_file.header['pixdim'][4] = 1
        ED_seg_file.header['sform_code'] = 1
        nib.save(ED_seg_file, start_time_datetime+"-"+"seg_la_4ch_ED.nii.gz")
        nib.save(ED_seg_file, os.path.join(patient_folder,"seg4_la_4ch_ED.nii.gz"))
        print("wrote to ",start_time_datetime+"-"+"seg_la_4ch_ED.nii.gz")