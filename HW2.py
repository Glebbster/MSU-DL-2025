import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from torch import nn
from torch import functional as F
# Creating a simple model using Seq
def create_model():
    model = nn.Sequential(nn.Linear(in_features = 784, out_features = 256 ),
                          nn.ReLU(),
                          nn.Linear(in_features= 256, out_features = 16),
                          nn.ReLU(),
                          nn.Linear(in_features = 16, out_features = 10),
                          )
    return model
model = create_model()
for param in model.parameters():
    nn.init.constant_(param, 1.)

assert torch.allclose(model(torch.ones((1, 784))), torch.ones((1, 10)) * 3215377.), 'Something is wrong with the model structure'

# Counting parameters in this model, model[0] is the first layer of the model, it needs to be taken separately to find out the number of weights in the first layer
def count_parameters(model):
    total = 0
    if isinstance(model, nn.Linear):
        total += model.weight.numel()
        if model.bias is not None:
            total += model.bias.numel()
    else:
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                total += layer.weight.numel()
                if layer.bias is not None:
                    total += layer.bias.numel()
    return total
print(count_parameters(model))

small_model = nn.Linear(128, 256)
assert count_parameters(small_model) == 128 * 256 + 256, 'Something is wrong, incorrect number of parameters'

medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
assert count_parameters(medium_model) == 128 * 32 + 32 * 10, 'Something is wrong, incorrect number of parameters'
print("Seems fine!")