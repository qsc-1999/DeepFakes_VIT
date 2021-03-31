import glob
import os
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import DeeperDataset,test_transforms

# load data
def load_test():
    test_root_dir = '/home/liu/deepfake_detection/VIT/data/FF++/test/c23/'
    test_root_dir = glob.glob(os.path.join(test_root_dir, '*'))
    data_test = []
    for i in range(len(test_root_dir)):
        test_split_dir = test_root_dir[i]
        #print(test_split_dir)
        #print('-----------------------')
        test_video_dir = glob.glob(os.path.join(test_split_dir, '*'))
        for j in range(len(test_video_dir)):
            test_video_list = glob.glob(os.path.join(test_video_dir[j],'*.png'))
            test_ = DeeperDataset(test_video_list, transform=test_transforms)
            data_test.append(test_)
    return data_test

# acc
def cal_acc(y, y_pred):
    y.cuda()
    y_pred.cuda()
    if len(y) != len(y_pred):
        print('their length is not equal!')
        return 0
    else:
        return (torch.eq(y, y_pred).sum() / len(y)).item()

# auc
def cal_auc(y, y_pred):
    # y:label [0,0,1,0,1]
    # y_pred: 预测为1时的概率 [:, 1]
    y.cuda()
    y_pred.cuda()
    if len(y) != len(y_pred):
        print('their length is not equal!')
        return 0
    else:
        return roc_auc_score(y,y_pred)

if __name__ == '__main__':
    model = torch.load('jx_vit_base_resnet50_384-9fd3c705.pth')
    #model.cuda()
    test_data = load_test()
    print(len(test_data))
    label_all = torch.tensor([])
    pred_all_prob = torch.tensor([])
    pred_all_cls = torch.tensor([])
    for i in range(len(test_data)):
        test_loader = DataLoader(dataset=test_data[i], batch_size=8, shuffle=True)
        for data, label in test_loader:
            data = data['image'].cuda()
            label = label.cuda()
            label_all += label
            output = model(data)
            output = F.softmax(output,dim=1)
            pred_all_prob += output[:,1]
            pred_all_cls += output.argmax(dim=1)
    AUC = cal_auc(label_all,pred_all_prob)
    ACC = cal_acc(label_all,pred_all_cls)




