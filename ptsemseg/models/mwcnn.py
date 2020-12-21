#from models import common
#from ptsemseg.models.utils import unetConv2, unetUp
from ptsemseg.models.common import *
import torch
import torch.nn as nn

'''
def make_model(args, parent=False):
    return mwcnn(args)'''

class mwcnn(nn.Module):
    def __init__(self, n_classes=19, conv=default_conv):
        super(mwcnn, self).__init__()
        print('TEST BIG TEST')
        n_resblocks = 20 #args.n_resblocks
        n_feats = 64 #args.n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = 3 #args.n_colors
        n_classes = 19

        act = nn.ReLU(True)

        self.DWT = DWT()
        self.IWT = IWT()

        n = 1
        m_head = [BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))


        d_l1 = [BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3 = []
        pro_l3.append(BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        i_l2 = [DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)]
        i_l2.append(BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))

        i_l1 = [DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1.append(BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))

        i_l0 = [DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        m_tail = [conv(n_feats, 19, kernel_size)]
        #m_tail = nn.Conv2d(n_feats, 19, 1)

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)
        #self.tail = m_tail

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
       # print("x_", x_.shape)
#         print("x", x.shape

#***********************************
#         x_avg = x.sum(1)/3
#         x_bg=x_avg.unsqueeze(1).repeat(1,19,1,1)
#         x = self.tail(self.i_l0(x_)) + x_bg 
#***********************************        
        x = self.tail(self.i_l0(x_))
#         print("tail",x.shape)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

