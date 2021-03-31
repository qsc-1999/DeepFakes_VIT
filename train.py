import time

import torch
import torch.nn as nn
from tqdm import tqdm
from config import load_config
from data import load_data
from model import Net
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import datetime

# auc曲线计算
# CrossEntropyLoss回顾
# softmax计算
# 数据增强方法
# git上传代码

def main(args):

    train_loader, valid_loader = load_data(args)
    model = Net()
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for data, label in tqdm(train_loader):
            data = data['image'].cuda()
            label = label.cuda()
            start = time.time()
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            #print(F.softmax(output,dim=1))
            acc = (F.softmax(output,dim=1).argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            optimizer.step()
            end = time.time()
            print(end-start)
            break
        lr_scheduler.step()
        writer.add_scalar('scalar/train_loss', epoch_loss, epoch)
        writer.add_scalar('scalar/train_acc', epoch_accuracy, epoch)
        print(
            f"Epoch : {epoch + 1}  - train_loss : {epoch_loss:.4f} - train_acc: {epoch_accuracy:.4f}\n"
        )

        if epoch % args.evaluation_epoch == 0:
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in tqdm(valid_loader):
                    data = data['image'].cuda()
                    label = label.cuda()
                    val_output = model(data)
                    val_loss = criterion(val_output, label)
                    acc = (F.softmax(val_output,dim=1).argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)
                print(
                    f"Epoch : {epoch + 1}  - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
                )
                if epoch_val_accuracy > 0.97:
                    torch.save(model, "./Model/model_FF++_HQ.pkl")
            writer.add_scalar('scalar/val_loss', epoch_val_loss, epoch)
            writer.add_scalar('scalar/val_acc', epoch_val_accuracy, epoch)

if __name__ == '__main__':
    args = load_config()
    main(args)