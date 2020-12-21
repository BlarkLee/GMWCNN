#from models import common
#from ptsemseg.models.utils import unetConv2, unetUp
from ptsemseg.models.common import *
import torch
import torch.nn as nn

'''
def make_model(args, parent=False):
    return mwcnn(args)'''

class gated_mwcnn(nn.Module):
    def __init__(self, n_classes=19, conv=default_conv):
        super(gated_mwcnn, self).__init__()
        n_resblocks = 20 #args.n_resblocks
        n_feats = 64 #args.n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = 3 #args.n_colors
        n_classes = 19
        
        # use offset whenever using db2
        self.offset = 0;
        
        C_1 = 2*n_feats
        C_2 = 2*C_1       # 4*n_feats
        C_3 = 2*C_2       # 8*n_feats
        
        act = nn.ReLU(True)
        
        # DWT, IWT operations (Haar)
        self.DWT = DWT()
        self.IWT = IWT()

        # DWT, IWT operations (Daubechies)
#         self.DWT = DWT(wavelet='db2')
#         self.IWT = IWT(wavelet='db2')

        # head operation preprocessing
        n = 1
        m_head = [BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = [DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        # ENCODERS
        # All have the form [BBlock, DBlock]        
        d_l1 = [BBlock(conv, 2*C_1 + self.offset, C_1, kernel_size, act=act, bn=False)]
        d_l1.append(DBlock_com1(conv, C_1, C_1, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(BBlock(conv, 2*C_2 + self.offset, C_2, kernel_size, act=act, bn=False))
        d_l2.append(DBlock_com1(conv, C_2, C_2, kernel_size, act=act, bn=False))
        
        d_l3 = []
        d_l3.append(BBlock(conv, 2*C_3 + self.offset, C_3, kernel_size, act=act, bn=False))
        d_l3.append(DBlock_com(conv, C_3, C_3, kernel_size, act=act, bn=False))
        
        # DECODERS
        # All have the form [DBlock, BBlock]        
        i_l3 = []
        i_l3.append(DBlock_inv(conv, C_3, C_3, kernel_size, act=act, bn=False))
        i_l3.append(BBlock(conv, C_3, 2*C_3 + self.offset, kernel_size, act=act, bn=False))
        # remember 2*C_3 = 4*C_2
        # 4C_2 -> IWT -> C_2

        i_l2 = [DBlock_inv1(conv, C_2, C_2, kernel_size, act=act, bn=False)]
        i_l2.append(BBlock(conv, C_2, 2*C_2 + self.offset, kernel_size, act=act, bn=False))

        i_l1 = [DBlock_inv1(conv, C_1, C_1, kernel_size, act=act, bn=False)]
        i_l1.append(BBlock(conv, C_1, 2*C_1 + self.offset, kernel_size, act=act, bn=False))

        # tail maps to the subset of classes we wish to classify
        i_l0 = [DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]
        m_tail = [conv(n_feats, n_classes, kernel_size)]
        
        # NOTE: see forward() for definition of track A/B
        # 1x1 cnn for transitioning between track A and track B
        # s stands for simple
        self.s_cnn_1 = nn.Sequential(nn.Conv2d(C_2, 1, 1))
        self.s_cnn_2 = nn.Sequential(nn.Conv2d(C_3, 1, 1))

        # encoder and decoder layers
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l3 = nn.Sequential(*d_l3)
        self.i_l4 = nn.Sequential(*i_l3)
        self.i_l5 = nn.Sequential(*i_l2)
        self.i_l6 = nn.Sequential(*i_l1)
        
        # reduction convolutions used in decoder in track A
        self.reduce_4_5 = nn.Sequential(nn.Conv2d(4*C_3 + self.offset, 2*C_3 + self.offset, kernel_size, padding=(1, 1)))
        self.reduce_5_6 = nn.Sequential(nn.Conv2d(4*C_2 + self.offset, 2*C_2 + self.offset, kernel_size, padding=(1, 1)))
        
        # new added gates for the bottom track of net (takes in output dimensionality of d_l1, d_l2)
        # NOTE: DWT increases by 4 in between gate1 and gate2
        self.gate1 = GatedSpatialConv2d(4*C_1 + self.offset, 2*C_2 + self.offset)
        self.gate2 = GatedSpatialConv2d(8*C_2 + self.offset, 2*C_3 + self.offset)
        
        # head layer
        self.head = nn.Sequential(*m_head)
        self.d_l0 = nn.Sequential(*d_l0)
        
        # tail layer
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)   

    def forward(self, x):
        # Please refer to the drawing for the following... 
        #
        # NOTE: In A_xn, B_xn, n refers to the output after the CNN layer and BEFORE the DWT!
        #          A_yn, n refers to the output after the CNN layer, concatentation, reduction, and IWT!       
        #         
        # A refers to original top track
        # B refers to new gated track    
        #         
        # x refers to the analysis (encorder) portion of the A,B track networks
        # y refers to the synthesis (decorder) portion of the A,B track networks

        # head layer
        A_x0 = self.d_l0(self.head(x))

        # Top track encoder result
        A_x1 = self.d_l1(self.DWT(A_x0))        
        A_x2 = self.d_l2(self.DWT(A_x1))
        A_x3 = self.d_l3(self.DWT(A_x2))
        
        # Bottom track encoder results
        B_x1 = self.gate1(self.DWT(A_x1), self.s_cnn_1(A_x2))
        B_x2 = self.gate2(self.DWT(B_x1), self.s_cnn_2(A_x3))
        
        # Add bottom track decoder to top track decoder
#         print('1: ' + str(A_x3.shape))
#         print('2: ' + str(self.i_l4(A_x3).shape))
#         print('3: ' + str(B_x2.shape))
#         print('4: ' + str(torch.cat((self.i_l4(A_x3), B_x2), 1).shape))
#         print('5: ' + str(self.reduce_4_5(torch.cat((self.i_l4(A_x3), B_x2), 1)).shape))
#         print('6: ' + str(self.IWT(self.reduce_4_5(torch.cat((self.i_l4(A_x3), B_x2), 1))).shape))
        A_y4 = self.IWT(self.reduce_4_5(torch.cat((self.i_l4(A_x3), B_x2), 1))) + A_x2
#         print(A_y4.shape)
        A_y5 = self.IWT(self.reduce_5_6(torch.cat((self.i_l5(A_y4), B_x1), 1))) + A_x1
        A_y6 = self.IWT(self.i_l6(A_y5)) + A_x0

        # tail layer
        y = self.tail(self.i_l0(A_y6))

        return y

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

