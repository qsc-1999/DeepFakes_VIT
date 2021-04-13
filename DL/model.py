import timm

def vit():
    model = timm.create_model('vit_base_patch16_384',pretrained=True)
    model.reset_classifier(2)
    print(model)
    return model

def Resvit():
    model = timm.create_model('vit_base_resnet50_384',pretrained=True)
    model.reset_classifier(2)
    print(model)
    return model

def ResNet():
    resnet = timm.create_model('resnetv2_50x1_bitm',pretrained=True)
    resnet.reset_classifier(2)
    print(resnet)
    return resnet







