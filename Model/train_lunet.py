import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # oppressing warnings


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import csv

import config
from dataset import get_dataloader
from encoder import PointNetfeat
from LUNet import LuNet

device = f'cuda:{config.GPU}'
torch.cuda.init()
def slice_tensor(x, start, end=None):
	if end < 0:
		y = x[:,start:,:,:]
	else:
		if end is None:
			end = start
		y = x[:,start:end + 1,:,:]

	return y

def compute_loss(pred, label):
    mask = slice_tensor(label, config.N_CLASSES + 1, -1)
    dist = slice_tensor(label, config.N_CLASSES, config.N_CLASSES)
    label  = slice_tensor(label, 0, config.N_CLASSES - 1)
    #print(label.shape)
    weight_norm = 2.0 * 3.0 ** 2.0
    weights_ce = 0.1 + 1.0 * torch.exp(- dist / weight_norm)
    weights_ce = weights_ce * mask
    loss = nn.CrossEntropyLoss()
    if config.FOCAL_LOSS :

        epsilon = 1.e-9
        gamma   = 2.
        pred_softmax  = F.softmax(pred, dim=1)
        #print(pred_softmax)
        cross_entropy = loss(label, pred_softmax)
        weights_fl = torch.mul(label, torch.pow(torch.sub(1., pred_softmax), gamma))
        weights_total = weights_fl * cross_entropy

        sum_weights = torch.sum(weights_total)
        weights_normalized = torch.div(weights_total, sum_weights + epsilon)

        # Compute the final weighted cross entropy loss
        loss = torch.sum(weights_normalized * cross_entropy)

    else:
        logits = pred.reshape(-1, config.N_CLASSES)
        label = label.reshape(-1, config.N_CLASSES)
        loss = F.cross_entropy(logits, label.argmax(dim=1))


    return loss

#y = lunet(points,neighbours)
    #y = torch.rand(config.BATCH_SIZE,config.N_CLASSES,config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    #loss = compute_loss(y, label)


def check_accuracy(loader, model):
    num_correct = 0
    accuracy = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for _ in range(config.VAL_ITER):
            for points, neighbours, label in loader:
                points = points.squeeze(0)
                neighbours = neighbours.squeeze(0)
                label  = slice_tensor(label.squeeze(0), 0, config.N_CLASSES - 1)

                preds = model(points, neighbours)
                pred_softmax = F.softmax(preds, dim=1)
                pred_class = torch.argmax(pred_softmax, dim=1)
                true_class = torch.argmax(label, dim=1)
                
                pred_class = pred_class.reshape(label.shape[0],config.N_CLASSES,-1)
                true_class = true_class.reshape( label.shape[0],config.N_CLASSES,-1)
                
                #print(pred_class.shape, true_class.shape)
                correct = torch.sum(pred_class == true_class)
                #print(correct)
                total = true_class.shape[0]*true_class.shape[1]*true_class.shape[2]
            accuracy = correct / total
            

    #print(
     #   f"Accuracy : {accuracy.item()/config.VAL_ITER}"
    #)
   
    model.train()
    return accuracy.item()#/config.VAL_ITER

    

def train(model,optim, loader, val_loader):
    acc = 0
    loss_dic = {}
    accuracy_dic = {}
    loop = tqdm(range(config.N_ITERS), initial=0,  desc='Training')
    for idx in loop:
        for batch_idx, (points, neighbours, label) in enumerate(loader):
            points = points.squeeze(0)#.cuda()
            neighbours = neighbours.squeeze(0)#.to(device)
            label = label.squeeze(0)#.to(device)
            
            y = model(points, neighbours)
            loss = compute_loss(y, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_dic[idx] = loss.item()
            
        acc = check_accuracy(val_loader, model)
        accuracy_dic[idx] = acc
        loop.set_postfix(loss = loss.item(), acc = acc)
        if(idx % config.ACC_FREQU == 0):
            
            #print("Saving information ...")
            
            torch.save(model.state_dict(), "./lunet.pth")
            with open('loss.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['index', 'loss'])
                for key, value in loss_dic.items():
                    writer.writerow([key, value])
            with open('acc.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['index', 'acc'])
                for key, value in accuracy_dic.items():
                    writer.writerow([key, value])
        
    print()
    print()
    print("Saving information ...")
    torch.save(model.state_dict(), "./lunet.pth")
    with open('loss.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'loss'])
        for key, value in loss_dic.items():
            writer.writerow([key, value])
    with open('acc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'acc'])
        for key, value in accuracy_dic.items():
            writer.writerow([key, value])

if __name__ == "__main__":

    points = torch.rand(config.BATCH_SIZE,5,1,config.IMAGE_HEIGHT*config.IMAGE_WIDTH)
    neighbours = torch.rand(config.BATCH_SIZE,5,8,config.IMAGE_HEIGHT*config.IMAGE_WIDTH)
    label = torch.rand(config.BATCH_SIZE,config.N_CLASSES+2,config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    
    lunet = LuNet(n_classes = config.N_CLASSES)
    opt = optim.Adam(lunet.parameters(),lr=config.LR)
    
    dataloader = get_dataloader(config.DATASET_PATH, config.BATCH_SIZE, config.NUM_WORKERS)
    val_loader = get_dataloader(config.VAL_PATH, config.BATCH_SIZE, config.NUM_WORKERS)

    train(lunet, opt,dataloader,val_loader)