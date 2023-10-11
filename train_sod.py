import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
import matplotlib.pyplot as plt
from dataset_sod import ImageFolder
import dataset_sod
from misc import AvgMeter, check_mkdir
from models.SSLNet import SSLNet
from torch.backends import cudnn
import torch.nn.functional as functional
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import math

cudnn.benchmark = True

torch.manual_seed(2023)
torch.cuda.set_device(0)


train_data = './data/RGBT_dataset/train'
test_data='./data/RGBT_dataset/val'






##########################hyperparameters###############################
ckpt_path = './SOD'
exp_name = 'SSLNet'
args = {
    'iter_num':20000,
    'train_batch_size': 10,
    'last_iter': 0,
    'lr': 1e-4,
    'lr_decay': 0.1,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'crop_size': 224,
    'snapshot': ''
}
##########################data augmentation###############################

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(args['crop_size'],args['crop_size']),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
#target_transform = transforms.ToTensor()
target_transform = transforms.Compose([
            transforms.Resize((args['crop_size'],args['crop_size'])),
            transforms.ToTensor()
            ])


##########################################################################

log_dir = os.path.join(ckpt_path, exp_name)
#summary_writer = SummaryWriter(log_dir)


train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform,args['crop_size'])
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)


#test_set = test_dataset(test_data,test_data,args['crop_size'])
#test_loader = DataLoader(test_set, batch_size=1, num_workers=12, shuffle=True)

#weights = torch.tensor([1.0,10.0,20.0])
criterion = nn.CrossEntropyLoss().cuda()
criterion_BCE = nn.BCEWithLogitsLoss().cuda()
criterion_MAE = nn.L1Loss().cuda()
criterion_MSE = nn.MSELoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
log_eval_path = os.path.join(ckpt_path, exp_name, 'evaluation' + '.txt')

def loss_fn(x, y):
	x = F.normalize(x, dim=1, p=2)
	y = F.normalize(y, dim=1, p=2)
	return 2 - 2 * (x * y).sum(dim=1)
 

class IOUBCE_loss(nn.Module):
    def __init__(self):
        super(IOUBCE_loss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs,targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b

IOUBCE = IOUBCE_loss().cuda()



def main():
    model = SSLNet()
	
    pretrained_dict = torch.load(os.path.join('./new_training/SSLNet_rgb/20000.pth'))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    net = model.cuda().train()
    
    
    
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])




    if len(args['snapshot']) > 0:
        print ('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']



    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)

def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record,loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 *args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            inputs, thermal, labels= data
            
            
           
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            thermal = Variable(thermal).cuda()
            labels = Variable(labels).cuda()
            
        
            labels[labels>0.5] = 1
            labels[labels!=1] = 0
        
            
            
            out,out2, out3,A5,A5_t = net(inputs,thermal)
            ##########loss#############
            optimizer.zero_grad()
            
            
            inputs1_flat = torch.mean(A5,dim=1).view(batch_size,-1)
            inputs2_flat = torch.mean(A5_t,dim=1).view(batch_size,-1)
            
            loss_ct = loss_fn(inputs1_flat,inputs2_flat).mean()
            lossa5 = IOUBCE(out_a5,labels)
            lossa5_t = IOUBCE(out_a5_t,labels)
            
            loss_mmhl = lossa5 + lossa5_t + loss_ct * 10
            
            loss1 = IOUBCE(out, labels)



                      
            
            total_loss = loss1+loss_mmhl

            
            total_loss.backward()
            
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(lossa5.item(), batch_size)
            loss3_record.update(lossa5_t.item(), batch_size)
            loss4_record.update(loss_ct.item(), batch_size)
            
            curr_iter += 1
           
            
            #############log###############
            if curr_iter %5000==0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                #torch.save(optimizer.state_dict(),
                           #os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
            log = '[iter %d], [total loss %.5f],[loss_5 %.5f],[loss_5t %.5f],[loss_ct %.5f],[lr %.13f]'  % \
                     (curr_iter, total_loss_record.avg, loss1_record.avg, loss2_record.avg, loss3_record.avg, loss4_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')
            #summary_writer.add_scalar('loss1', loss1.item(), curr_iter)
            #summary_writer.add_scalar('loss2', loss2.item(), curr_iter)
            
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return
            #############end###############
            
        #summary_writer.close()
        
if __name__ == '__main__':
    main()
