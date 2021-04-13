import time
import torch
import torch.nn as nn
from tqdm import tqdm
from config import load_config
from data import load_data
from model import vit,Resvit,ResNet
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import datetime

def main(args):

    train_loader, valid_loader = load_data(args)
    model = ResNet()
    #model = nn.DataParallel(model)
    model.cuda()
    #params = dense_egde_vit_params(model, args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        #print(epoch)
        for data, label in tqdm(train_loader):
            #continue
            data = data.cuda()
            label = label.cuda()
            #print(data)
            #print(label)
            #start = time.time()
            # Log = LoG(data,window,9)
            output = model(data)
            #print(output)
            optimizer.zero_grad()
            loss = criterion(output, label)
            #print(loss)
            loss.backward()
            #print(F.softmax(output,dim=1).argmax(dim=1))
            #print("-------------------------------------------")
            #print(output.argmax(dim=1))
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            optimizer.step()
            #end = time.time()
            #print(end-start)
            #break
        lr_scheduler.step()
        # writer.add_scalar('scalar/train_loss', epoch_loss, epoch)
        # writer.add_scalar('scalar/train_acc', epoch_accuracy, epoch)
        print(
            f"Epoch : {epoch + 1}  - train_loss : {epoch_loss:.4f} - train_acc: {epoch_accuracy:.4f}\n"
        )
        #break

        if epoch % args.evaluation_epoch == 0:
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in tqdm(valid_loader):
             #       continue
                    #data = data['image'].cuda()
                    data = data.cuda()
                    label = label.cuda()
                    val_output = model(data)
                    val_loss = criterion(val_output, label)
                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)
                print(
                    f"Epoch : {epoch + 1}  - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
                )
                # if epoch_val_accuracy > 0.90:
                #     torch.save(model, "./Model/model_FF++_HQ_"+str(epoch_val_accuracy.item())+".pkl")
                # writer.add_scalar('scalar/val_loss', epoch_val_loss, epoch)
                # writer.add_scalar('scalar/val_acc', epoch_val_accuracy, epoch)

if __name__ == '__main__':
    args = load_config()
    main(args)
