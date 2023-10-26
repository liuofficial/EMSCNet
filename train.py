import os
import argparse
import numpy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
from branch import branch as create_branch
import torchvision.transforms.functional as TF
import random
from typing import Sequence
from vit_model import vit_base_patch16_224 as create_model

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #storage
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label,test_images_path,test_images_label \
            ,ect_index  = read_split_data(args.data_path,val_rate=0.0,test_rate=0.8,p=args.seed)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.25,1.0), interpolation=TF.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            MyRotateTransform([0, 90, 180, 270]),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([transforms.Resize(int(img_size * 1.143), TF.InterpolationMode.BICUBIC),
                                   transforms.CenterCrop(img_size),transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}


    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"]
                              )
    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"]
                             )



    batch_size = args.batch_size
    print("batch_size",batch_size)
    nw=0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)
    #B
    model_B = create_model(num_classes=args.num_classes).cuda()
    #G
    model_G = create_model(num_classes=args.num_classes).cuda()

    branch=create_branch(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # Delete the weight of the category
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model_B.load_state_dict(weights_dict, strict=False))
        print(model_G.load_state_dict(weights_dict, strict=False))

    train_list = nn.ModuleList()
    train_list.append(model_B)
    train_list.append(branch)
    train_list.cuda()

    optimizer = optim.SGD([{'params': train_list.parameters(), 'lr': 0.001}
                            ],momentum=0.9,  weight_decay=5e-5)


    def adjust_learning_rate(optimizer,epoch):
        if epoch==20:
            lr=0.0001
            optimizer.param_groups[0]['lr'] = lr
        elif epoch==90:
            lr=0.00001
            optimizer.param_groups[0]['lr'] = lr

    t_acc=0.0
    for epoch in range(args.epochs):
        #train
        train_one_epoch(model_B=model_B,model_G=model_G,ect_index=ect_index,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,branch=branch,
                                                epoch=epoch,args=args
                                                )
        #test
        if (epoch>30):
            test_acc = evaluate(model=model_B,
                                           data_loader=test_loader,
                                           device=device,
                                           epoch=epoch,ect_index=ect_index,args=args,branch=branch)
            if t_acc < test_acc:
                t_acc = test_acc
                print("test_maxacc=", test_acc)
        #保存模型
        #torch.save(model.state_dict(), "weights/model-{}.pth".format(epoch))
        adjust_learning_rate(optimizer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--num_positive', type=int, default=10)
    parser.add_argument('--num_negative', type=int, default=400)

    parser.add_argument('--weights', type=str,
                        default='vit_base_patch16_224.pth.baiduyun.p.downloading',
                        help='initial weights path')

    parser.add_argument('--temperature', default='10', type=float)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="../data/AID")
    opt = parser.parse_args()
    main(opt)