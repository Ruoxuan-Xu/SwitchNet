import torch
import numpy as np
import random
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision
choose_mask_size = 3
img_size = 32
seed = 123
threshold = 0.5

save_path = './masks/mask_size_'+str(choose_mask_size)+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

mask = torch.zeros(1,img_size,img_size)
UAE = torch.zeros(3,img_size,img_size)


random_w_number = np.random.randint(0, img_size-choose_mask_size+1)
random_h_number = np.random.randint(0, img_size-choose_mask_size+1)

mask[:,random_w_number:random_w_number+3,random_h_number:random_h_number+3] = threshold*torch.rand(1,3,3)
UAE[:,random_w_number:random_w_number+3,random_h_number:random_h_number+3] = threshold*torch.randn(3,3,3)

torch.save(mask,save_path+'mask_seed_'+str(seed))
torch.save(UAE,save_path+'UAE_seed_'+str(seed))

combine_img=255*(UAE.cpu().data.numpy()[::-1,:,:].transpose(1,2,0)+0.5)
Image.fromarray(np.uint8(combine_img)).convert('RGB').save(save_path+'UAE_seed_'+str(seed)+'.png')
combine_img=255*(mask.repeat(3,1,1).cpu().data.numpy().transpose(1,2,0))
Image.fromarray(np.uint8(combine_img)).convert('RGB').save(save_path+'mask_seed_'+str(seed)+'.png')