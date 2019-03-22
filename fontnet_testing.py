#this file is used for testing , depends on weights from both autoencoder and fontnet_tr

from __future__ import division, print_function, unicode_literals
import numpy as np
#import tensorflow
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
from io import BytesIO
import util2 as ut
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

#hyperparameters
#===============================================
batch_size = 45 #changing the might result in some discrepancies in test function
num_epochs = 2
learning_rate = 0.02
use_gpu = True
input_size = 101
#================================================

#this is test loader
class AdVFR_test(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, root_dir, transform=None):
                
        # root_dir  - the root directory of the dataset
        # train     - a boolean parameter representing whether to return the training set or the test set
        # transform - the transforms to be applied on the images before returning them
    
        self.dataStore, self.labelStore = ut.read_bcf_file(root_dir,2)
        self.transform = transform
        self.crop_img = []
        self.crop_label = 0
        
    def __len__(self):
        return self.dataStore.size() * 15
    
    def crop_test(self,image):
        images = []
        crop_images=[]
        for i in range(3):
            rand = np.random.uniform(1.5,3.5)
            width = int(image.width*rand)
            height = image.height
            images.append(image.resize((width,height)))
            randin = np.random.uniform(0,width-height,5)
            for j in range(5):
                dim = (int(randin[j]),0,int(randin[j])+height,height)
                crop_images.append(images[i].crop(dim))
        return crop_images    
    
    def __getitem__(self, idx):
        # idx - the index of the sample requested
        image_id = int(idx / 15)
        crop_image_id = idx % 15        
        if ( crop_image_id == 0):
            orig_image = Image.open(BytesIO(self.dataStore.get(image_id)))
            orig_label = int(self.labelStore[image_id])
            #print(orig_image.size)
            
            self.crop_img = self.crop_test(orig_image)
            self.crop_label = orig_label
            image = self.crop_img[0]
        else:
            image = self.crop_img[crop_image_id]
        image = image.convert('L')    
        label = self.crop_label
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    
composed_transform = transforms.Compose([transforms.Scale((input_size,input_size)), 
                                         transforms.ToTensor()])
test_dataset = AdVFR_test(root_dir='.', transform=composed_transform) # Supply proper root_dir
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#=============================================================
class Fontnet_SCAE(nn.Module): # Extend PyTorch's Module class
    def __init__(self):
        super(Fontnet_SCAE, self).__init__() # Must call super __init__()
        
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,stride=2,bias=True,padding=2)  # 64, 51, 51
        self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2,stride=2,return_indices=True) # 128, 27, 27
        self.conv2 = nn.Conv2d(64, 128,kernel_size=3, stride=2,bias=True, padding=1)  #  128, 13, 13
        self.bn2 = nn.BatchNorm2d(128)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.maxpool2 = nn.MaxPool2d(2,stride=2)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)  # 64, 25, 25
        self.unpool1 = nn.MaxUnpool2d(2,stride=2)  #u 64, 51, 51
        self.deconv2 = nn.ConvTranspose2d(64,1,5,stride=2,padding=2) # 1,101,101
#         self.unpool2 = nn.MaxUnpool2d(3,stride=2)
        
    def forward(self, x):
#start of encoder        
        org_size = x.size()    # 1x101x101
        out = self.conv1(x)    # 64x51x51
        out = self.bn1(out)    # 64x51x51  
        MU1size = out.size()
#         out = self.relu(out)
        out,p_indices = self.maxpool1(out)      #64x25x25
#         print('maxp1',out.size())
        out = self.conv2(out)             #128x25x25
#         print('conv2',out.size())
        out1 = self.bn2(out)              #128x13x13
#         print('bn2',out1.size())
#         out1 = self.relu2(out)
#end of encoder
#start of decoder
        out = self.deconv1(out1)          #128x25x25 
#         print('deconv1',out.size())
#         print(MU1size)
        out = self.unpool1(out,p_indices,output_size=MU1size)     #128x51x51                 
#         print('unpool1',out.size())
        out = self.deconv2(out)          #128x101x101              
#         print('deconv2',out.size())
        #out = self.unpool2(out,output_size=org_size)
        #print('unpool2',out.size())
#end of decoder        
#         out_en = self.maxpool2(out1) 
        out_en = out1
        
        return out,out_en
    
    
model_scae = Fontnet_SCAE()
model_scae.load_state_dict(torch.load('MLProj_scae_0.pkl', map_location=lambda storage, loc: storage)) 
if(torch.cuda.is_available() and use_gpu):
    model_scae.cuda()
    
#=========================================================================
class Fontnet(nn.Module): # Extend PyTorch's Module class
    def __init__(self,num_classes=2383):
        super(Fontnet, self).__init__() # Must call super __init__()
        # inp
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,stride=1,bias=True,padding=1)
        self.conv4 = nn.Conv2d(256,256,kernel_size=3,stride=1,bias=True,padding=1)
        self.conv5 = nn.Conv2d(256,256,kernel_size=3,stride=2,bias=True,padding=1)
        
        self.fc6 = nn.Linear(12544,4096) # 256x7x7
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.6)
        self.fc7 = nn.Linear(4096,num_classes)
        
    def forward(self, x):
        
        out = self.conv3(x) 
        out = self.conv4(out)                
        out = self.conv5(out)  
#         print(out.size())
        
        out = self.fc6(out.view(-1,12544))
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc7(out.view(-1,4096))        #fc - Linear
#         print(out.size())
        return out

model = Fontnet(num_classes = 2383) # 100 classes since CIFAR-100 has 100 classes
model.load_state_dict(torch.load('MLProj_fontnet_1.pkl', map_location=lambda storage, loc: storage)) # Supply the path to the weight file
if(torch.cuda.is_available() and use_gpu):
    model.cuda()
#==================================================================

def test(model,model_scae):
    total = 0
    correct = 0
    count = 0
    accum_score = np.zeros((3,2383))
    for images, labels in test_loader:
        images = Variable(images)
        
        if(use_gpu):
            images = images.cuda()
        
        _,out = model_scae(images)
        outputs = model(out)
        #outputs is of Variable datatype contains the final classifications #use: print(outputs.data)
        #You guys can play around with this and modify the code below it as you like
        outputs_np = outputs.data.cpu().numpy() # .cpu() moves the tensor from GPU to CPU , .numpy converts tensors to numpy
        accum_score[0] = np.sum(outputs_np[:15],axis=0)
        accum_score[1] = np.sum(outputs_np[15:30],axis=0)
        accum_score[2] = np.sum(outputs_np[30:45],axis=0)
        print(np.argmax(accum_score[0]),labels[0])
        print(np.argmax(accum_score[1]),labels[15])
        print(np.argmax(accum_score[2]),labels[30])
        count=0
        #score, predicted = torch.max(accum_score, 1)
        #total += labels.size(0)
        #correct += (predicted.cpu() == labels.cpu()).sum()
        accum_score=np.zeros((3,2383))
    print('Accuracy of the network on the %d test images: %d %%' % (len(test_dataset),100 * correct / total))
    
test(model,model_scae)