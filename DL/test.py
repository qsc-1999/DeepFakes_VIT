import glob
import os
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import DeeperDataset,test_transforms
import numpy as np

# load data
def load_test():
    test_root_dir = '/home/liu/qsc/VIT/data/FF++/test/c23/'
    test_root_dir = glob.glob(os.path.join(test_root_dir, '*'))
    data_test = []
    for i in range(len(test_root_dir)):
    #    if i==1 or i==2 or i==3:
    #        continue
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
    #y.cuda()
    #y_pred.cuda()i
    #y = [1,1]
    #y_pred = [0,1]
    #print(y.type())
    #print(y_pred.type())
    if len(y) != len(y_pred):
        print('their length is not equal!')
        return 0
    else:
        return np.equal(y, y_pred).sum() / len(y)

# auc
def cal_auc(y, y_pred):
    # y:label [0,0,1,0,1]
    # y_pred: 预测为1时的概率 [:, 1]
    #y.cuda()
    #y_pred.cuda()
    #y = [1,0]
    #y_pred = [1,0]
    #print(y)
    if len(y) != len(y_pred):
        print('their length is not equal!')
        return 0
    else:
        return roc_auc_score(y,y_pred)

if __name__ == '__main__':
    #print(cal_acc())
    #print(cal_auc())
    model = torch.load('./Model/model_FF++_HQ_0.9723833799362183.pkl')
    #model = torch.nn.DataParallel(model)
    model.cuda()
    test_data = load_test()
    print(len(test_data))
    #label_all = torch.tensor([]).cpu()
    #pred_all_prob = torch.tensor([]).cpu()
    #pred_all_cls = torch.tensor([]).cpu()
    label_all = []
    pred_all_prob = []
    pred_all_cls = []
    k=0
    for i in range(len(test_data)):
        print(k)
        k+=1
        #print(len(test_data[i]))
        test_loader = DataLoader(dataset=test_data[i], batch_size=20, shuffle=True)
        for data, label in test_loader:
            data = data.cuda(1)
            label = label.cuda(1)
            #print(label.shape)
            #label_all = torch.cat((label_all,label.cpu()))
            label_all += label.tolist()
            #print(len(label_all))
            output = model(data)
            output = F.softmax(output,dim=1)
            pred_cls = output.argmax(dim=1).tolist()
            pred_prob = output[:,1].tolist()
            #print(output[:,1].float8())
            pred_all_prob += pred_prob
            #pred_all_prob = torch.cat((pred_all_prob,pred_prob.cpu()))
            pred_all_cls += pred_cls
            #pred_all_cls = torch.cat((pred_all_cls,output.argmax(dim=1).cpu()))
    print("enter cal")
    AUC = cal_auc(label_all,pred_all_prob)
    ACC = cal_acc(label_all,pred_all_cls)
    print("AUC = ",AUC)
    print("ACC = ",ACC)




