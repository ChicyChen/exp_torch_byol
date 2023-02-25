import numpy as np
import torch
import torch.nn as nn



class Fine_Tune(nn.Module):
    def __init__(self, model, input_dim=512, class_num=51):
        super(Fine_Tune, self).__init__()
        self.input_dim = input_dim
        self.encoder = model
        self.encoder.eval()
        self.linear_pred = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, class_num)
        )

    def forward(self, block):
        h = self.encoder.get_representation(block) # need to refered by BYOL
        output = self.linear_pred(h)
        return output
    

    
    
    