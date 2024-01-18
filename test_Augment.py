import scipy.io as sio
from torch import nn
import torch
import argparse
from material import *
import numpy as np
import utils
from torch.autograd import Variable
import class_model
import autoagement as aa
import general_augment as ga

parser = argparse.ArgumentParser("test_Augment")
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--atlas_size', type=int, default=90, help='90,160,200')
parser.add_argument('--drop1', type=float, default=0.2)
parser.add_argument('--drop2', type=float, default=0.3)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--step_size', type=int, default=50000)
parser.add_argument('--epoch', type=int, default=2000)

args = parser.parse_args()


def get_batch(X_input):
    X_output = torch.utils.data.DataLoader(X_input, batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                           num_workers=0)
    return X_output


def test_main(list_ind):
    load_data = sio.loadmat('/home/ai/data/wangxingy/data/ADHD/ALLADHD1_NETFC_SG_Pear.mat')
    tr_x = torch.tensor(load_data['net_train']).view(-1, 1, args.atlas_size, args.atlas_size).type(torch.FloatTensor)
    ts_x = torch.tensor(load_data['net_test']).view(-1, 1, args.atlas_size, args.atlas_size).type(torch.FloatTensor)
    tr_y = torch.tensor(load_data['Y_train'].reshape(-1), dtype=torch.float64)
    ts_y = torch.tensor(load_data['Y_test'].reshape(-1), dtype=torch.float64)
    if args.atlas_size == 90:
        list_altas = list_aal
    elif args.atlas_size == 160:
        list_altas = list_dos
    else:
        list_altas = list_cc
    au_tr_x = aa.augment(list_ind, list_altas, tr_x)
    au_tr_y = np.tile(tr_y, 5)
    X_train = get_batch(au_tr_x)
    Y_train = get_batch(au_tr_y)
    X_test = get_batch(ts_x)
    Y_test = get_batch(ts_y)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = class_model.BrainNetCNN(args.atlas_size, args.drop1, args.drop2)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.epoch):
        train_loss, train_acc = train(X_train, Y_train, model, optimizer, criterion)
        print('epoch = ', epoch, ' train_loss = ', train_loss, 'train_acc = ', train_acc)
        test_loss, test_acc, test_sen, test_spe, = infer(X_test, Y_test, model, criterion)
        print(' test_loss = ', test_loss, 'test_acc = ', test_acc, 'test_sen = ', test_sen, 'test_spe = ', test_spe)
        scheduler.step()


def train(X_train, Y_train, model, optimizer, criterion):
    loss_AM = utils.AvgrageMeter()
    acc_AM = utils.AvgrageMeter()
    for steps, (input, target) in enumerate(zip(X_train, Y_train)):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=True).cuda()
        target = Variable(target, requires_grad=True).cuda()
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target.long())
        loss.backward()
        optimizer.step()
        accuracy = utils.accuracy(logits, target)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
    return loss_AM.avg, acc_AM.avg


def infer(X_vaild, Y_vaild, model, criterion):
    loss_AM = utils.AvgrageMeter()
    acc_AM = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(zip(X_vaild, Y_vaild)):
        with torch.no_grad():
            input = Variable(input).cuda()
        with torch.no_grad():
            target = Variable(target).cuda()

        logits = model(input)
        loss = criterion(logits, target.long())

        accuracy = utils.accuracy(logits, target)
        sen, spe = utils.sensitivity(logits, target)
        n = input.size(0)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)

    return loss_AM.avg, acc_AM.avg, sen, spe


list_ind = [4, 5, 1.0, 0.0, 0, 1, 0.17, 0.0, 5, 1, 1.0, 0.14, 1, 4, 1.0, 0.36]
test_main(list_ind)