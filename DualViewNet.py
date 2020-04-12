import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
from torch import tensor
class MyDropOut(nn.Module):
    def __init__(self,p=0.5):
        super(MyDropOut, self).__init__()
        self.p = p
        self.dropout1 = nn.Dropout(self.p)
        self.dropout2 = nn.Dropout(self.p)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.dropout2(x)
        return x

    def __repr__(self):
            return self.__class__.__name__ +"dropout"

class DualViewNet(nn.Module):
    def __init__(self,depth=7,num_classes=5):
        super(DualViewNet, self).__init__()
        self.base1 = EfficientNet.from_pretrained('efficientnet-b{}'.format(depth),num_classes=num_classes)
        self.base2 = EfficientNet.from_pretrained('efficientnet-b{}'.format(depth),num_classes=num_classes)
        self._dropout = MyDropOut()
        self._fc = nn.Linear(self.base1._fc.in_features*2, self.base1._global_params.num_classes)
        self._avg_pooling = self.base1._avg_pooling
    def load_state_dict_first(self, state_dict, strict=True):
        self.base1.load_state_dict(state_dict,strict=False)
        self.base2.load_state_dict(state_dict,strict = False)
    def forward(self, x):
        bs = x.size(0)
        x1 = x[:,0:3,:,:]
        x2 = x[:,3:6,:,:]
        x1 = self.base1.extract_features(x1)
        x2 = self.base2.extract_features(x2)
        x = torch.cat([x1,x2],axis=1)
        x =self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x