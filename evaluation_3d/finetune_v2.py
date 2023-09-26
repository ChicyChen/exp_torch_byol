import numpy as np
import torch
import torch.nn as nn



class Fine_Tune(nn.Module):
    def __init__(self, model, input_dim=512, class_num=101, dropout=0.5, num_layer=1):
        super(Fine_Tune, self).__init__()
        self.input_dim = input_dim
        self.encoder = model
        self.encoder.eval()
        if num_layer == 1:
            self.linear_pred = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, class_num)
            )
        else:
            self.linear_pred = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, class_num)
            )

        self._initialize_weights(self.linear_pred)

    def forward(self, block):
        h = self.encoder(block) 
        output = self.linear_pred(h)
        return output
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
    
    

    
    
    