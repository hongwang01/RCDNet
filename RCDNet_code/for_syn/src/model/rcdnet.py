# A Model-driven Deep Neural Network for Single Image Rain Removal
# https://arxiv.org/abs/2005.01333
#http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf

from model import common
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as  F
import os
from torch.autograd import Variable
import torch.distributions.laplace
import scipy.io as io
import numpy as np

def make_model(args, parent=False):
    return Mainnet(args)

# rain kernel  C initialized by the Matlab code "init_rain_kernel.m"
kernel = io.loadmat('./init_kernel.mat')['C9'] # 3*32*9*9
kernel = torch.FloatTensor(kernel)

# filtering on rainy image for initializing B^(0) and Z^(0), refer to supplementary material(SM)
w_x = (torch.FloatTensor([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])/9)
w_x_conv = w_x.unsqueeze(dim=0).unsqueeze(dim=0)

class Mainnet(nn.Module):
    def __init__(self, args):
        super(Mainnet,self).__init__()
        self.S = args.stage                                      #Stage number S includes the initialization process
        self.iter = self.S-1                                     #not include the initialization process
        self.num_M = args.num_M
        self.num_Z = args.num_Z

        # Stepsize
        self.etaM = torch.Tensor([1])                                   # initialization
        self.etaX = torch.Tensor([5])                                   # initialization
        self.eta1 = nn.Parameter(self.etaM, requires_grad=True)         # usd in initialization process
        self.eta2 = nn.Parameter(self.etaX, requires_grad=True)         # usd in initialization process
        self.eta11 = self.make_eta(self.iter, self.etaM)                # usd in iterative process
        self.eta12 = self.make_eta(self.iter, self.etaX)                # usd in iterative process

        # Rain kernel
        self.weight0 = nn.Parameter(data=kernel, requires_grad = True)  # used in initialization process
        self.conv = self.make_weight(self.iter, kernel)                 # rain kernel is inter-stage sharing. The true net parameter number is (#self.conv /self.iter)

        # filter for initializing B and Z
        self.w_z_f0 = w_x_conv.expand(self.num_Z, 3, -1, -1)
        self.w_z_f = nn.Parameter(self.w_z_f0, requires_grad=True)

        # proxNet in initialization process
        self.xnet = Xnet(self.num_Z+3)                                  # 3 means R,G,B channels for color image
        self.mnet = Mnet(self.num_M)
        # proxNet in iterative process
        self.x_stage = self.make_xnet(self.S, self.num_Z + 3)
        self.m_stage = self.make_mnet(self.S, self.num_M)
        # fine-tune at the last
        self.fxnet = Xnet(self.num_Z + 3)

        # for sparse rain layer
        self.f = nn.ReLU(inplace=True)
        self.taumm = torch.Tensor([1])
        self.tau = nn.Parameter(self.taumm, requires_grad=True)
    def make_xnet(self, iters, channel):
        layers = []
        for i in range(iters):
            layers.append(Xnet(channel))
        return nn.Sequential(*layers)
    def make_mnet(self, iters, num_M):
        layers = []
        for i in range(iters):
            layers.append(Mnet(num_M))
        return nn.Sequential(*layers)
    def make_eta(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1)
        eta = nn.Parameter(data=const_f, requires_grad = True)
        return eta
    def make_weight(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1,-1,-1,-1)
        weight = nn.Parameter(data=const_f, requires_grad = True)
        return weight
            
    def forward(self, input):
        # save mid-updating results
         ListB = []
         ListCM = []

        # initialize B0 and Z0 (M0 =0)
         z0 = F.conv2d(input, self.w_z_f, stride =1, padding = 1)              # dual variable z with the channels self.num_Z
         input_ini = torch.cat((input, z0), dim=1)
         out_dual = self.xnet(input_ini)
         B0 = out_dual[:,:3,:,:]
         Z = out_dual[:,3:,:,:]

         # 1st iteration: Updating B0-->M1
         ES = input - B0
         ECM = self.f(ES-self.tau)                                            #for sparse rain layer
         GM = F.conv_transpose2d(ECM, self.weight0/10, stride=1, padding=4)   # /10 for controlling the updating speed
         M = self.m_stage[0](GM)
         CM = F.conv2d(M, self.conv[1,:,:,:,:]/10, stride =1, padding = 4)    # self.conv[1,:,:,:,:]：rain kernel is inter-stage sharing
       
         # 1st iteration: Updating M1-->B1
         EB = input - CM
         EX = B0-EB
         GX = EX
         x_dual = B0-self.eta2/10*GX
         input_dual = torch.cat((x_dual, Z), dim=1)
         out_dual = self.x_stage[0](input_dual)
         B = out_dual[:,:3,:,:]
         Z = out_dual[:,3:,:,:]
         ListB.append(B)
         ListCM.append(CM)

         for i in range(self.iter):
             # M-net
             ES = input - B
             ECM = CM- ES
             GM = F.conv_transpose2d(ECM,  self.conv[1,:,:,:,:]/10, stride =1, padding = 4)
             input_new = M - self.eta11[i,:]/10*GM
             M = self.m_stage[i+1](input_new)

             # B-net
             CM = F.conv2d(M,  self.conv[1,:,:,:,:]/10, stride =1, padding = 4)
             ListCM.append(CM)
             EB = input - CM
             EX = B - EB
             GX = EX
             x_dual = B - self.eta12[i,:]/10*GX
             input_dual = torch.cat((x_dual,Z), dim=1) 
             out_dual  = self.x_stage[i+1](input_dual)
             B = out_dual[:,:3,:,:]
             Z = out_dual[:,3:,:,:]
             ListB.append(B)
         out_dual = self.fxnet(out_dual)                # fine-tune
         B = out_dual[:,:3,:,:]
         ListB.append(B)
         return B0, ListB, ListCM

# proxNet_M
class Mnet(nn.Module):
    def __init__(self, channels):
        super(Mnet, self).__init__()
        self.channels = channels
        self.tau0 = torch.Tensor([0.5])
        self.taum= self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels,-1,-1)
        self.tau = nn.Parameter(self.taum, requires_grad=True)                      # for sparse rain map
        self.f = nn.ReLU(inplace=True)
        self.resm1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation =1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                   )
        self.resm2 = nn.Sequential(nn.Conv2d(self.channels, self.channels,kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resm3 = nn.Sequential(nn.Conv2d(self.channels, self.channels,kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resm4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                   )
    def forward(self, input):
        m1  = F.relu(input + self.resm1(input))
        m2  = F.relu(m1+ self.resm2(m1))
        m3  = F.relu(m2+ self.resm3(m2))
        m4  = F.relu(m3+ self.resm4(m3))
        m_rev =self.f(m4-self.tau)                                     # for sparse rain map
        return m_rev

# proxNet_B
class Xnet(nn.Module):
    def __init__(self, channels):
        super(Xnet, self).__init__()
        self.channels = channels
        self.resx1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                 nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels,  kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
    def forward(self, input):
        x1  = F.relu(input + self.resx1(input))
        x2  = F.relu(x1 + self.resx2(x1))
        x3  = F.relu(x2 + self.resx3(x2))
        x4  = F.relu(x3 + self.resx4(x3))
        return x4
