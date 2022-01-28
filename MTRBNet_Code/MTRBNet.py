# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F



class MTRBNet(nn.Module):
	def __init__(self):
		super(MTRBNet,self).__init__()

		self.en = En_Decoder(3,8)
         
	def forward(self,x):
        
		eout = self.en(x)
      
		return eout

class En_Decoder(nn.Module):
	def __init__(self,inchannel,channel):
		super(En_Decoder,self).__init__()
        


		self.el = MTRB(channel)
		self.em = MTRB(channel*2)
		self.es = MTRB(channel*4)
		self.ds = MTRB(channel*4)
		self.dm = MTRB(channel*2)
		self.dl = MTRB(channel)
        
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   

		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  

        
		self.conv_in = nn.Conv2d(inchannel,channel,kernel_size=1,stride=1,padding=0,bias=False)        
		self.conv_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)    
    		

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')

	def forward(self,x):
        
		x_elin = self.conv_in(x)

		elout = self.el(x_elin)
        
		x_emin = self.conv_eltem(self.maxpool(elout))
        
		emout = self.em(x_emin)
        
		x_esin = self.conv_emtes(self.maxpool(emout))        
        
		esout = self.es(x_esin)
        
		dsout = self.ds(esout)
        
		x_dmin = self._upsample(self.conv_dstdm(dsout),emout) + emout
        
		dmout = self.dm(x_dmin)

		x_dlin = self._upsample(self.conv_dmtdl(dmout),elout) + elout
        
		dlout = self.dl(x_dlin)
        
		x_out = self.conv_out(dlout)

        
		return x_out
    
class MTRB(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(MTRB,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel*1,  channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_2_1 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_2 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)

        
		self.conv_3_1 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3_2 = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3_3 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)

 
		self.conv_4_1 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_2 = nn.Conv2d(channel*3,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_3 = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_4 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_5_1 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_2 = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_3 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)


		self.conv_6_1 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_6_2 = nn.Conv2d(channel*3,channel,kernel_size=3,stride=1,padding=1,bias=False)

		self.conv_7_1 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)

        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.sig = nn.Sigmoid()

		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')


	def forward(self,x):
        
		x_1_1 = self.act(self.norm(self.conv_1_1(x)))

		x_2_1 = self.act(self.norm(self.conv_2_1(x_1_1)))
		x_2_2 = self.act(self.norm(self.conv_2_2(torch.cat((x_2_1 , x_1_1),1))))

        
		x_3_1 = self.act(self.norm(self.conv_3_1(x_2_1)))
		x_3_3 = self.act(self.norm(self.conv_3_3(x_2_2)))
		x_3_2 = self.act(self.norm(self.conv_3_2(torch.cat((x_3_1 , x_3_3 , x_2_1 , x_2_2),1))))
        
        
		x_4_1 = self.act(self.norm(self.conv_4_1(x_3_1)))
		x_4_4 = self.act(self.norm(self.conv_4_4(x_3_3)))
		x_4_2 = self.act(self.norm(self.conv_4_2(torch.cat((x_4_1 , x_3_1 , x_3_2),1))))
		x_4_3 = self.act(self.norm(self.conv_4_3(torch.cat((x_4_2 , x_4_4 , x_3_2 , x_3_3),1))))
 
    
		x_5_1 = self.act(self.norm(self.conv_5_1(torch.cat((x_4_1 , x_4_2),1))))
		x_5_3 = self.act(self.norm(self.conv_5_3(torch.cat((x_4_3 , x_4_4),1))))
		x_5_2 = self.act(self.norm(self.conv_5_2(torch.cat((x_5_1 , x_5_3 , x_4_2 , x_4_3),1))))
     
        
		x_6_1 = self.act(self.norm(self.conv_6_1(torch.cat((x_5_1 , x_5_2),1))))
		x_6_2 = self.act(self.norm(self.conv_6_2(torch.cat((x_6_1 , x_5_2 , x_5_3),1))))


		x_7_1 = self.act(self.norm(self.conv_7_1(torch.cat((x_6_1 , x_6_2),1))))

        
		x_out = self.act(self.norm(self.conv_out(x_7_1)) + x)


		return	x_out
        