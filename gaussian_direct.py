from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from math import *
import torch.nn.functional as F

class kernel_conv(object):
    """
    Args:
        r(int): ratio of downsapling
    output:
        img: PIL.Image
    """
    def __init__(self, kernel, channels, kernel_size):
        super(kernel_conv, self).__init__()
        # kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.kernel_size=kernel_size
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        print(self.weight)
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """

    if 'Normalize' in str(transform_train):
        # mean and var of transform
        # filter arr: filter(function,iterable)，function -- judge function，iterable -- iterable object
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3: #rgb
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1: #gray
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_

class T_gaussian(object):
    """
    add blur for rgb/gray img
    Args:
        sigma(float): the var of gauss
        alpha(float): the ratio between width and length of gauss,best smaller than 1.0
        theta(float): the angle of gauss
        kernel_size(float): the kernel_size of gaussian
    output:
        img: PIL.Image
    """
    def __init__(self, sigma=1.0, alpha=1.0, theta=90, kernel_size=5):
        self.sigma = sigma
        self.alpha = alpha
        self.theta = theta
        self.kernel_size = kernel_size

    #Main Body
    def __call__(self, img): 
        # Initializing value of x,y as grid of kernel size
        # in the range of kernel size
        x, y = np.meshgrid(np.linspace(-self.kernel_size, self.kernel_size, self.kernel_size),
                          np.linspace(-self.kernel_size, self.kernel_size, self.kernel_size))
        dst_x = np.sqrt((x*cos(self.theta)+y*sin(self.theta))**2)
        dst_y = np.sqrt((-x*sin(self.theta)+y*cos(self.theta))**2)
        # lower normal part of gaussian
        normal = (1/sqrt(2.0 * np.pi * self.sigma**2))*(1/sqrt(2.0 * np.pi * (self.sigma*self.alpha)**2))
        # Calculating Gaussian filter
        gauss = np.exp(-((dst_x)**2 / (2.0 * self.sigma**2))-((dst_y)**2 / (2.0 * (self.sigma*self.alpha)**2))) * normal
        gauss = gauss/gauss.sum()
        # using torch as conv tool
        T = transforms.Compose([
          transforms.ToTensor()])
        img_ = T(img).unsqueeze(0)
        channels=img_.size()[1]
        F = kernel_conv(gauss, channels, self.kernel_size)
        img_ = F.forward(img_).squeeze(0)
        # img_ = img_.squeeze(0)
        img_ =transform_invert(img_ ,T)
        img_=np.array(img_)
        return img_


norm_mean = [0.485, 0.456, 0.406] 
norm_std = [0.229, 0.224, 0.225]  
train_transform = transforms.Compose([
    T_gaussian(3,0.5,45,5),
    transforms.ToTensor(), 
    transforms.Normalize(norm_mean, norm_std),
])



path_img = "./117355.png"  
img = Image.open(path_img).convert('RGB') 
img_tensor = train_transform(img) 


convert_img = transform_invert(img_tensor, train_transform)

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(figsize=(6,2.5)) 
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(convert_img) 
plt.show()
plt.savefig('./pr.png')
plt.close()
