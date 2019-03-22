# this is first part of the network which learns the compressed representation of the images
#produces a model_scae which is used in fontnet_tr 

from __future__ import division, print_function, unicode_literals
import numpy as np
import tensorflow
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
#import matplotlib.pyplot as plt

#Hyperparameters
#================================================
batch_size = 45
num_epochs = 2
learning_rate = 0.02
use_gpu = True
input_size = 101      # Don't change input size
#==================================================
class SCAE_train(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, root_dir, transform=None):
                
        # root_dir  - the root directory of the dataset
        # train     - a boolean parameter representing whether to return the training set or the test set
        # transform - the transforms to be applied on the images before returning them

        self.dataStore,_ = ut.read_bcf_file(root_dir,0)
        self.dstoresize = self.dataStore.size()
        self.transform = transform
        self.real_list = []
        for file in os.listdir("scrape-wtf-new"):
            self.real_list.append(os.path.join("scrape-wtf-new", file))
        
    def __len__(self):
        return self.dataStore.size() * 3
        
    def crop_train(self,image):
        image = image.resize((int((image.width*120)/image.height),120))
        if(image.width<image.height):
            #return image.resize((120,120))
            return image.resize((120,120))
        randint = int(np.random.uniform(0,image.width-120))
        dim = (randint,0,randint+120,120)
        cropped_image=image.crop(dim)
        #plt.imshow(cropped_image)
        return cropped_image
    
    def __getitem__(self, idx):
        # idx - the index of the sample requested
        randint=np.random.uniform(0,1)
        if(randint<0.9):
            image_id = idx % self.dstoresize
            orig_image = Image.open(BytesIO(self.dataStore.get(image_id)))
            randidx=int(np.random.uniform(0,len(self.real_list)))
            orig_image = Image.open(self.real_list[randidx]).convert('L')
        else:
            randidx=int(np.random.uniform(0,len(self.real_list)))
            orig_image = Image.open(self.real_list[randidx]).convert('L')
        
        image = self.crop_train(orig_image)
        
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    
composed_transform = transforms.Compose([transforms.Scale((input_size,input_size)), 
                                         transforms.ToTensor()])
scae_dataset = SCAE_train(root_dir='.', transform=composed_transform) # Supply proper root_dir
scae_loader = torch.utils.data.DataLoader(dataset=scae_dataset, batch_size=batch_size, shuffle=True)


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
#model.load_state_dict(torch.load('CIFAR-100_weights')) # Supply the path to the weight file
if(torch.cuda.is_available() and use_gpu):
    model_scae.cuda()
    
scae_criterion = nn.MSELoss()
scae_optimizer = torch.optim.Adam(model_scae.parameters(), lr=learning_rate, weight_decay=1e-5)

def scae_train():
    for epoch in range(num_epochs):
        for i, images in enumerate(scae_loader):  
            # Convert torch tensor to Variable
            images = Variable(images)
            if(use_gpu):
                images=images.cuda()
                
            # ====================== forward pass ==================
            outputs,_ = model_scae(images)
            loss = scae_criterion(outputs, images)
            
            #=========================backward pass ===================
            
            scae_optimizer.zero_grad()  # zero the gradient buffer
            loss.backward()
            scae_optimizer.step()
            
            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(scae_dataset)//batch_size, loss.data[0]))
        
        
        # change the filename and save the model , Remember to put the same file name to upload weights in other network
        filename =  'MLProj_scae_'+str(epoch)+'.pkl'
        torch.save(model_scae.state_dict(),filename)

scae_train()