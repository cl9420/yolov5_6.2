import torch
import torch.nn as nn
m = nn.AdaptiveAvgPool2d((None,1))
m1 = nn.AdaptiveAvgPool2d((1,None))
m2 = nn.AdaptiveAvgPool2d(1)
input = torch.randn(1,1,8, 9)
output = m(input)
output1 = m1(input)
output2 = m2(input)
print(output)
print(output1)
print('nn.AdaptiveAvgPool2d((5,1)):',output.shape)
print('nn.AdaptiveAvgPool2d((None,5)):',output1.shape)
print('nn.AdaptiveAvgPool2d(1):',output2.shape)
