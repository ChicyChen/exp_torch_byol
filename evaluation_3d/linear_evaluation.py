import numpy as np
import torch
import torch.nn as nn



class Linear_Eval(nn.Module):
    def __init__(self, model, input_dim=512, class_num=101, dropout=0.2):
        super(Linear_Eval, self).__init__()
        self.input_dim = input_dim
        self.encoder = model
        self.encoder.eval()
        self.linear_pred = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, class_num)
        )

    def forward(self, block):
        # print(block.size())
        with torch.no_grad():
            h = self.encoder.get_representation(block) # need to refered by BYOL
        # print(torch.min(h), torch.max(h))
        output = self.linear_pred(h)
        # print(torch.min(output), torch.max(output))
        return output
    

    
    
    