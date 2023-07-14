import torch

# Consistent model to SlowFast or the follwing works
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)

# https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/models/resnet.html

# https://pytorch.org/vision/0.12/_modules/torchvision/models/video/resnet.html

# https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
# https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/resnet.py