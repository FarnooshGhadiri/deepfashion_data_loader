import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiLabelModel(nn.Module):
    def __init__(self, basemodel, basemodel_output, num_classes):
        super(MultiLabelModel, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes
        for index, num_class in enumerate(num_classes):
            setattr(self, "FullyConnectedLayer_" + str(index), nn.Linear(basemodel_output, num_class))
    
    def forward(self, x):
        x = self.basemodel.forward(x)
        outs = list()
        dir(self)
        for index, num_class in enumerate(self.num_classes):
            fun = eval("self.FullyConnectedLayer_" + str(index))
            out = fun(x)
            if index==0:
             b = F.softmax(out, dim=1)
            else:
             a = F.sigmoid(out)

        return torch.cat([a, b], dim=1)

def LoadPretrainedModel(model, pretrained_state_dict):
    model_dict = model.state_dict()
    union_dict = {k : v for k,v in pretrained_state_dict.iteritems() if k in model_dict}
    model_dict.update(union_dict)
    return model_dict

def BuildMultiLabelModel(basemodel, basemodel_output, num_classes):
    return MultiLabelModel(basemodel, basemodel_output, num_classes)
