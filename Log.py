# encoding=gbk
import torch
import torch.nn.functional as F
from torch.autograd import Variable

#将最后的矩阵中的元素归一化到0~1之间
def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min)/(max-min)

#LoG变换
def LoG(img1_tensor,window,window_size):
    channel = img1_tensor.size()[1]
    window = Variable(window.expand(channel, 1, window_size, window_size).contiguous())
    output = F.conv2d(img1_tensor, window, padding = window_size//2, groups = channel)
    output = minmaxscaler(output)# 归一化到0~1之间
    return output

#近似卷积核
window = torch.Tensor([[[0,1,1,2,2,2,1,1,0],
                        [1,2,4,5,5,5,4,2,1],
                        [1,4,5,3,0,3,5,4,1],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [2,5,0,-24,-40,-24,0,5,2],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [1,4,5,3,0,3,4,4,1],
                        [1,2,4,5,5,5,4,2,1],
                        [0,1,1,2,2,2,1,1,0]]]).cuda()

