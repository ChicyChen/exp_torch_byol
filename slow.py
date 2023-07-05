import torch

# Consistent model to SlowFast or the follwing works
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)