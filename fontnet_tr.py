#This file does the supervised learning of the fontnet. it depends on the weights provided by autoencoder.py

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

#Hyperparameters
#=============================================
batch_size = 45
num_epochs = 2
learning_rate = 0.02
use_gpu = True
input_size = 101 # Don't change this
#=================================================

class AdVFR_train(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, root_dir, transform=None):
                
        # root_dir  - the root directory of the dataset
        # train     - a boolean parameter representing whether to return the training set or the test set
        # transform - the transforms to be applied on the images before returning them

        self.dataStore, self.labelStore = ut.read_bcf_file(root_dir,0)
        self.dstoresize = self.dataStore.size()
        self.transform = transform
        
    def __len__(self):
        return self.dataStore.size() * 5
        
    def crop_train(self,image):
        randint = int(np.random.uniform(0,image.width-image.height))
        dim = (randint,0,randint+image.height,image.height)
        cropped_image=image.crop(dim)
        return cropped_image
    
    def __getitem__(self, idx):
        # idx - the index of the sample requested
        image_id = idx % self.dstoresize
        orig_image = Image.open(BytesIO(self.dataStore.get(image_id)))
        #image = image.convert('RGB')
        image = self.crop_train(orig_image)
        label = int(self.labelStore[image_id])
    
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    
composed_transform = transforms.Compose([transforms.Scale((input_size,input_size)), 
                                         transforms.ToTensor()])
train_dataset = AdVFR_train(root_dir='.', transform=composed_transform) 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


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
#================================
#load the weights file u got from Autoencoder.py
model_scae.load_state_dict(torch.load('MLProj_scae_0.pkl', map_location=lambda storage, loc: storage)) # Supply the path to the weight file

#====================================
if(torch.cuda.is_available() and use_gpu):
    model_scae.cuda()  # Just putting the network inside GPU    
    
class Fontnet(nn.Module): # Extend PyTorch's Module class
    def __init__(self,num_classes=2383):
        super(Fontnet, self).__init__() # Must call super __init__()
        # inp
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,stride=1,bias=True,padding=1)
        self.conv4 = nn.Conv2d(256,256,kernel_size=3,stride=1,bias=True,padding=1)
        self.conv5 = nn.Conv2d(256,256,kernel_size=3,stride=1,bias=True,padding=1)
        
        self.fc6 = nn.Linear(43264,4096) # 256x7x7
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.6)
        self.fc7 = nn.Linear(4096,4096)
        self.dropout2 = nn.Dropout(0.7)        
        self.fc8 = nn.Linear(4096,num_classes)
        
    def forward(self, x):
        
        out = self.conv3(x) 
        out = self.conv4(out)                
        out = self.conv5(out)  
#         print(out.size())
        
        out = self.fc6(out.view(-1,43264))
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc7(out.view(-1,4096))        #fc - Linear
        out = self.dropout2(out)
        out = self.fc8(out)        #fc - Linear

        return out



model = Fontnet(num_classes = 51) # To Manan: here we put number of font classes we want as output. 2383 is original # of classes 
#model.load_state_dict(torch.load('CIFAR-100_weights')) # Supply the path to the weight file
if(torch.cuda.is_available() and use_gpu):
    model.cuda()
    
    
# Loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


def train():
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Convert torch tensor to Variable
            images = Variable(images)
            labels = Variable(labels)
            if(use_gpu):
                images=images.cuda()
                labels=labels.cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            
            _,images = model_scae(images)
            #print(images.size())
                
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        #put the file name you want to save, this weights file will be used inside training 
        filename =  'MLProj_fontnetsm_'+str(epoch)+'.pkl'
        torch.save(model.state_dict(),filename)

train()