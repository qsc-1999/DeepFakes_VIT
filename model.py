import timm
import torch

def Net():
    model = timm.create_model('vit_base_resnet50_384', pretrained=True)
    model.head = torch.nn.Linear(768,2)
    print(model)
    return model
