import wandb
import yaml
from tqdm import tqdm

import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def calc_iou(pred, gt):
    pred = (pred[0] > .5).to(torch.float32)
    gt = gt[0][0].to(torch.float32)

    assert pred.shape == (320, 320)
    assert gt.shape == (320, 320)

    iou = (pred * gt).sum() / torch.maximum(pred, gt).sum()
    iou = iou.detach().cpu().numpy().item()
    return iou

def main():
    with open('./wandb_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    run = wandb.init(config=config, project='haimi')

    bce_loss = nn.BCELoss(size_average=True)

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = bce_loss(d0,labels_v)
        loss1 = bce_loss(d1,labels_v)
        loss2 = bce_loss(d2,labels_v)
        loss3 = bce_loss(d3,labels_v)
        loss4 = bce_loss(d4,labels_v)
        loss5 = bce_loss(d5,labels_v)
        loss6 = bce_loss(d6,labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

        return loss0, loss


    # ------- 2. set the directory of training dataset --------

    model_name = 'u2net' #'u2netp'

    # data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    train_data_dir = "/sailhome/ksarge/viscam/bike_symbols/clean_data_dir/train/"
    test_data_dir = "/sailhome/ksarge/viscam/bike_symbols/clean_data_dir/test/"

    tra_image_dir = ''
    tra_label_dir = ''

    image_ext = '__rgb.png'
    iel = len(os.path.splitext(image_ext)[0])
    label_ext = '__mask.png'

    model_dir = os.path.join("/sailhome/ksarge/viscam/bike_symbols/", 'saved_models', model_name + os.sep)

    epoch_num = 100000
    batch_size_train = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0

    def get_label_lists(data_dir):

        tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

        tra_lbl_name_list = []
        for img_path in tra_img_name_list:
            img_name = img_path.split(os.sep)[-1]

            aaa = img_name.split(".")
            bbb = aaa[0:-1]
            imidx = bbb[0]
            for i in range(1,len(bbb)):
                imidx = imidx + "." + bbb[i]

            tra_lbl_name_list.append(data_dir + tra_label_dir + imidx[:-iel] + label_ext)
        return tra_img_name_list, tra_lbl_name_list

    tra_img_name_list, tra_lbl_name_list = get_label_lists(train_data_dir)
    test_img_name_list, test_lbl_name_list = get_label_lists(test_data_dir)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    test_salobj_dataset = SalObjDataset(img_name_list = test_img_name_list,
                                        lbl_name_list = test_lbl_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    debug = True
    if debug:
        print("debugging!!")
        save_frq = 10 # save the model every 2000 iterations
        eval_max = 10
        log_img_every = 3
    else:
        save_frq = 2000
        eval_max = 1e6
        log_img_every = 20

    cur_step = 0
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            cur_step += 1

            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            wandb.log({
                'epoch': epoch, 
                'loss': running_loss / ite_num4val,
            }, step=cur_step)

            if ite_num % save_frq == 0:
                # run eval:
                print("run eval")
                net.eval()

                ious = []
                log_img_dicts = []
                
                for i, data in tqdm(enumerate(test_salobj_dataloader)):
                    inputs, labels = data['image'], data['label']
                    inputs = inputs.type(torch.FloatTensor)
                    labels = labels.type(torch.FloatTensor)
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
                    d1,*_= net(inputs_v)

                    # normalization
                    pred = d1[:,0,:,:]
                    pred = normPRED(pred)
                    
                    iou = calc_iou(pred, labels_v)
                    ious.append(iou)

                    if i % log_img_every == 0:
                        pred = pred.detach().cpu().numpy()
                        pred = pred[0, ..., None]

                        inputs_v = inputs_v.detach().cpu().numpy()
                        inputs_v = inputs_v[0].transpose((1,2,0))

                        labels_v = labels_v.detach().cpu().numpy()
                        labels_v = labels_v[0]
                        
                        log_img_dict = {
                            f"pred_{i}": pred,
                            f"rgb_{i}": inputs_v,
                            f"gt_{i}": labels_v,
                        }
                        log_img_dicts.append(log_img_dict)

                    if i > eval_max:
                        break
                        
                wandb_images = {
                    key: wandb.Image(log_img_dict[key], caption=key)
                    for log_img_dict in log_img_dicts
                    for key in log_img_dict
                }
                wandb.log({
                    'test_iou': np.mean(ious),
                    **wandb_images
                }, step=cur_step)
                
                print("saving model")
                os.makedirs(model_dir, exist_ok=True)
                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0

                net.train()  # resume train
                ite_num4val = 0

if __name__ == '__main__':
    main()