import json
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

def read_split_data(root: str, val_rate: float = 0.01,test_rate: float=0.8,p=0):
    random.seed(p)  # 保证随机结果可复现
    print("random=",p)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    every_class_train=[]
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_test_path = random.sample(images, k=int(len(images) * (val_rate+test_rate)))

        val_path = random.sample(val_test_path, k=int(len(images) * val_rate))

        test_path=[]
        for img_p in val_test_path:
            if img_p not in val_path:
                test_path.append(img_p)

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            elif img_path in test_path:
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
        every_class_train.append(int(len(images)-len(images) * (val_rate+test_rate)))#96.92
        #every_class_train.append(40)
    ect_index=np.zeros(len(flower_class)+1,dtype=int)
    for i in range(1,len(every_class_train)+1):
        ect_index[i]=int(ect_index[i-1]+every_class_train[i-1])
    print("ect_index",ect_index)
    print("{} every_class_num were found in the dataset.".format(every_class_train))
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for test.".format(len(test_images_path)))
    return train_images_path, train_images_label, test_images_path,test_images_label,ect_index

def clip_gradient(optimizer, grad_clip=0.5):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


from distill_loss.KD import kd
from distill_loss.logsum import logsum


@torch.no_grad()
def momentum_update_key_encoder(encoder_q,encoder_k,m:float=0.999):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

def train_one_epoch(model_B, optimizer, data_loader, device, epoch,model_G,ect_index,branch,args):
    model_B.train()
    branch.train()
    loss_function = torch.nn.CrossEntropyLoss()
    multi_loss=logsum(args)
    loss_kd=kd(args).train()

    total_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0

    data_loader = tqdm(data_loader)
    if epoch==0:#Filling the queue
        momentum_update_key_encoder(model_B, model_G,m=0)
        for step, data in enumerate(data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                _,queue1,queue2,features = model_G(images)
            queue1 = queue1.detach()[:ect_index[args.num_classes]]
            queue2 = queue2.detach() [:ect_index[args.num_classes]]

            for i, label in enumerate(labels):  # feed image1
                begin = ect_index[label]
                end = ect_index[label + 1]
                position = torch.argmin(queue2[begin:end]) + begin
                queue1[position] = features[i].detach()
                queue2[position] = step
    else:
        for step, data in enumerate(data_loader):
            # 原始优化器
            images, labels = data
            images=images.to(device)

            labels=labels.to(device)
            sample_num += images.shape[0]
            predict_1,queue1,queue2,features= model_B(images)

            pred_classes1 = torch.max(predict_1, dim=1)[1]
            accu_num += torch.eq(pred_classes1, labels.to(device)).sum()
            loss = loss_function(predict_1, labels)

            with torch.no_grad():
                queue1=queue1.detach()[:ect_index[args.num_classes]]
                queue2=queue2.detach()[:ect_index[args.num_classes]]

                pairwised1 = torch.ones(features.shape[0],queue1.shape[0])
                pairwised  = torch.ones(features.shape[0],queue1.shape[0])

                for i,label in enumerate(labels):
                    beg=ect_index[label]
                    en=ect_index[label+1]
                    # negative
                    pairwised1[i,beg:en]=-1000
                    # positive
                    pairwised[i,:beg]=1000
                    pairwised[i,en:]=1000
                _, pos_negative = torch.topk(pairwised1, k=1900, dim=1)
                _, pos_positive = torch.topk(pairwised, k=40,largest=False, dim=1)

                #random selection of samples
                p1=torch.randperm(1900)[:args.num_negative]
                p2=torch.randperm(40)[:args.num_positive]
                pos_negative=pos_negative[:,p1]
                pos_positive=pos_positive[:,p2]
                feautre_negative=queue1[pos_negative].detach()
                feautre_positive=queue1[pos_positive].detach()

            multi_feature,predict_2=branch(features,feautre_negative.detach(),feautre_positive.detach())
            loss += multi_loss(multi_feature)
            loss += loss_function(predict_2, labels)
            loss += loss_kd(predict_1,predict_2)

            with torch.no_grad():
                _, features_G, _, _ = model_G(images)
                momentum_update_key_encoder(model_B,model_G,0.999)#momentum update encoder_G
                for i,label in enumerate(labels):#queue update
                    begin=ect_index[label]
                    end=ect_index[label+1]
                    #get the oldest position
                    position=torch.argmin(queue2[begin:end])+begin
                    queue1[position]=features_G[i]
                    #first in first out mark
                    queue2[position]=step+epoch*10000

            loss.backward()
            clip_gradient(optimizer)
            total_loss += loss.detach()

            data_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                                   total_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            optimizer.step()
            optimizer.zero_grad()
    return


@torch.no_grad()
def evaluate(model, data_loader, device, epoch,ect_index,args,branch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    branch.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    total_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images=images.cuda()
        sample_num += images.shape[0]
        predict,_,_,_= model(images)
        pred_classes = torch.max(predict, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(predict, labels.to(device))
        total_loss += loss
        data_loader.desc = "[test epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                                  total_loss.item() / (step + 1),
                                                                                      (accu_num.item()) / (sample_num))


    return (accu_num.item()) / (sample_num)