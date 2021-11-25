import help_functions_local as hf
import boto3
import pydicom
from pydicom.filebase import DicomBytesIO
import requests
import torch
import os
import sys
import json
import shutil


sys.path.append('D:\\Work\\MRI\\git_BASH\\outside-repos\\Attention-Gated-Networks\\')
sys.path.append('D:\\Work\\MRI\\git_BASH\\Machine-Learning-Utils')
sys.path.append('D:\\Work\\MRI\\MRI Engine-20211110T135657Z-001\\MRI Engine\\ukbb_cardiac\\common')
import ModelArchitectures
import models.layers.grid_attention_layer as grid_attention_layer
import MyUtilFunctions
import importlib
class Unet_Attention(torch.nn.Module):     
    def __init__(self,**kwargs):
        super(Unet_Attention,self).__init__()
        self.input_shape = kwargs["input_shape"]
        self.num_classes = kwargs["num_classes"]
        self.filter_sizes=[self.input_shape[0]] + [64,128,256,512,1024]
        
        self.downblocks = torch.nn.ModuleList( [self.downsamplingBlock(self.filter_sizes[i],self.filter_sizes[i+1]) \
                           for i in range(len(self.filter_sizes)-1)] )
        
        self.reversed_filters = self.filter_sizes[::-1][:-1] + [self.num_classes]
        #self.reversed_filters = [2*self.filter_sizes[i] for i in range(len(self.filter_sizes)-1,0,-1)] + [self.num_classes]
        self.upblocks = torch.nn.ModuleList( [self.upsamplingBlock(self.reversed_filters[i]+self.reversed_filters[i+1],self.reversed_filters[i+1]) \
                        for i in range(len(self.reversed_filters)-2)] )
        self.classification_layer = torch.nn.Conv2d(self.reversed_filters[-2],self.reversed_filters[-1],1)
        self.attentionblocks = torch.nn.ModuleList( [grid_attention_layer.GridAttentionBlock2D(self.filter_sizes[i],self.filter_sizes[i+1],sub_sample_factor=2) for i in range(1,len(self.filter_sizes)-1)] )
        self.apply(self.init_weights) 

    #@staticmethod
    def downsamplingBlock(self,in_filters,out_filters):
        return torch.nn.ModuleList([torch.nn.Conv2d(in_filters,out_filters,3,padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm2d(out_filters),
                                  torch.nn.Conv2d(out_filters,out_filters,3,padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm2d(out_filters)
                                  ] )
    
    #@staticmethod
    def upsamplingBlock(self,in_filters,out_filters):
        '''return torch.nn.Sequential(torch.nn.Upsample(scale_factor=2),
                                  torch.nn.Conv2d(in_filters,out_filters,3,padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm2d(out_filters),
                                  torch.nn.Conv2d(out_filters,out_filters,3,padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm2d(out_filters))'''
        return torch.nn.ModuleList([
                                  torch.nn.Conv2d(in_filters,out_filters,3,padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm2d(out_filters),
                                  torch.nn.Conv2d(out_filters,out_filters,3,padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm2d(out_filters)])

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

     #def forward(self,features):
     #   code = self.encode(features)
     #   recon = self.decode(code)
     #   return recon
    def forward(self,features):
        #print("in model shape ",features.shape)
        #for p in self.parameters():
        #    print("device ",p.device)
        segmentation = features
        prevconv = []
        for block_index,d in enumerate(self.downblocks):
            if block_index != 0:
                segmentation = torch.nn.functional.max_pool2d(segmentation,2)
            for i in range(0,len(d)):
                segmentation = d[i](segmentation)
            prevconv.append(segmentation)
        #print("len prevconv",len(prevconv))
        for block_index,u in enumerate(self.upblocks):
            #print(block_index)
            attention_out = self.attentionblocks[-1-block_index](prevconv[len(prevconv)-2-block_index],prevconv[len(prevconv)-1-block_index])
            segmentation = u[0](torch.cat([torch.nn.Upsample(scale_factor=2)(segmentation),attention_out[0]],dim=1))
            for i in range(1,len(u)):
                segmentation = u[i](segmentation)
        
        '''features = self.downblock1(features)
        features = self.downblock2(features)
        features = self.upblock1(features)
        features = self.upblock2(features)
        '''
        
        '''features = self.downblockparam_0(features)
        features = self.downblockparam_1(features)
        features = self.downblockparam_2(features)
        features = self.downblockparam_3(features)
        features = self.upblockparam_0(features)
        features = self.upblockparam_1(features)
        features = self.upblockparam_2(features)
        features = self.upblockparam_3(features)
        '''
        segmentation = self.classification_layer(segmentation)
        return segmentation
            
    def encode(self,features):
        for d in self.downblocks:
            features = d(features)
        return features
    
    def decode(self,code):
        for u in self.upblocks:
            code = u(code)
        return code


session = boto3.Session(
                        aws_access_key_id='AKIAX6HXN2UPZYZVMD72',
                        aws_secret_access_key='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',
                    )

s3 = session.client('s3')
s3_resource = session.resource('s3')
bucket_name = 'viewer.dyadmed.com'
mf = hf.MRIFunctions()

# ############################local
# dicom_dir = 'D:\\Work\\MRI\\MRI Engine-20211110T135657Z-001\\MRI Engine\\test_dicom\\'
# folder_list = os.listdir(dicom_dir)
# dcm_img_list = []
# for folder in folder_list:
#     try:
#         dicom_list = os.listdir(dicom_dir + folder)
# #         print(dicom_list)
#     except:
#         continue
#     else:
#         for file in dicom_list:
#             try:
#                 dataset = pydicom.dcmread(dicom_dir + folder + '//' + file)
#             except:
#                 continue
#             else:
#                 dcm_img_list.append(dataset)

########################S3
mri_study_id = '1.2.826.0.1.497891220150426162617'
mf.get_all_instance_ids_and_source(mri_study_id)
instance_src_and_id_list = mf.get_all_instance_ids_and_source(mri_study_id)
print(len(instance_src_and_id_list))
dcm_img_list = []
for source_id, src in instance_src_and_id_list:
    if src ==1: # 'S3'
        try:
            aws_dir = mf.get_aws_dir_for_instance(source_id)
            dicom_obj = s3_resource.Object(bucket_name, aws_dir)
            dicomdata = dicom_obj.get()['Body'].read()
        except Exception as e:
            print(e)
            continue 
        try:
            dicombytes = DicomBytesIO(dicomdata)
            dataset = pydicom.dcmread(dicombytes)
            #print('Getting data from S3')
        except Exception as e:
            print(e)
            continue     
    elif src == 2: # 'PACS'
        try:
            username = 'dyadadmin'
            password = 'Dyadmed2021'
            pacs_id = mf.get_PACS_ID_for_instance(source_id)
            instance_content = requests.get("http://3.216.207.22:8043/instances/" + str(pacs_id) + "/file", \
                auth=(username, password))
                # 'D:/Work/EchoCardium/EchoTriage_Demo/Pipeline_for_PACS_AWS_DB/dicom_tmp/Instance.dicom')
            #print(instance_content.status_code)
            instance_file_path = 'Instance.dicom'
        except Exception as e:
            print(e)
            continue
        with open(instance_file_path, 'wb') as f:
            f.write(instance_content.content)
        try:
            dataset = pydicom.dcmread(instance_file_path)
            #print('Getting data from PACS')
        except Exception as e:
            print(e)
            continue
    else:
        continue
    dcm_img_list.append(dataset)
print("dcm_img_list length = ", len(dcm_img_list))
unet_model = Unet_Attention(input_shape=(1,256,256),num_classes=4)
state_dict_path = 'D:\\Work\\MRI\\models\\seg_model_with_unet_attention_20200929-220810-403495_epoch_5_stoppedearly.pt'
unet_model.load_state_dict(torch.load(state_dict_path))

seg_model = ModelArchitectures.Unet(num_classes=6,input_shape=(1,256,256))
state_dict_path = 'D:\\Work\\MRI\\models\\lax_model_20210224-110516-923019_epoch_5.pt'
seg_model.load_state_dict(torch.load(state_dict_path))


sa_4d_vol, position_to_name, name_to_position, nifty_img = mf.create_sax_vol(dcm_img_list)
la_4d_vol, position_to_name_la, name_to_position_la, nifty_img_la = mf.create_lax_vol(dcm_img_list)
print(sa_4d_vol.shape, la_4d_vol.shape)

json_dic, prefix = mf.create_json(dcm_img_list, unet_model, seg_model)
json.dump(json_dic,open("test_out_trt.json","w"))

for path in os.listdir('./'):
    if path.startswith(prefix):
        try:
            os.remove(path)
        except:
            shutil.rmtree(path)
