import torch
import numpy as np
import random
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision

choose_mask_size = 3
num_random_pixels = choose_mask_size ** 2
img_size = 32
seed = 123
threshold = 0.2

save_path = './masks/mask_size_'+str(choose_mask_size)+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

mask = torch.zeros(1, img_size, img_size)
UAE = torch.zeros(3, img_size, img_size)

# Generate unique random positions
random_positions = set()
while len(random_positions) < num_random_pixels:
    random_w_number = np.random.randint(0, img_size)
    random_h_number = np.random.randint(0, img_size)
    random_positions.add((random_w_number, random_h_number))

# Assign random values to those positions
for (w, h) in random_positions:
    mask[:, w, h] = threshold * torch.rand(1)
    UAE[:, w, h] = threshold * torch.randn(3)

torch.save(mask, save_path + 'mask_seed_' + str(seed))
torch.save(UAE, save_path + 'UAE_seed_' + str(seed))

combine_img = 255 * (UAE.cpu().data.numpy()[::-1, :, :].transpose(1, 2, 0) + 0.5)
Image.fromarray(np.uint8(combine_img)).convert('RGB').save(save_path + 'UAE_seed_' + str(seed) + '.png')
combine_img = 255 * (mask.repeat(3, 1, 1).cpu().data.numpy().transpose(1, 2, 0))
Image.fromarray(np.uint8(combine_img)).convert('RGB').save(save_path + 'mask_seed_' + str(seed) + '.png')
