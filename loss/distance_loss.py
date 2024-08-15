import torch
import torch.nn as nn

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, cam1, cam2):
        """
        计算两张灰度级CAM图像之间的欧氏距离，并返回负值作为损失
        """
        # Flatten the CAM images to 1D tensors
        cam1_flat = cam1.view(-1)
        cam2_flat = cam2.view(-1)
        
        # Compute the Euclidean distance
        distance = torch.sqrt(torch.sum((cam1_flat - cam2_flat) ** 2))
        
        # Return the negative distance
        return -distance