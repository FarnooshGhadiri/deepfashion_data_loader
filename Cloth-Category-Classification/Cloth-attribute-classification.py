import os
import sys
import time
import copy
import random
import logging
import numpy as np
import torch

print "Pytorch Version: ", torch.__version__
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict, defaultdict
from data.dataset_fashion import DeepFashionDataset
from models.model import load_model, save_model, modify_last_layer_lr
from options.options import Options
from util import util
from torch.utils.data import DataLoader
from util.webvisualizer import WebVisualizer
from torch.utils.data.sampler import SubsetRandomSampler

#approach like this
#  https://www.ritchievink.com/blog/2018/04/12/transfer-learning-with-pytorch-assessing-road-safety-with-computer-vision/

def forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, phase):
    if opt.cuda:
        inputs = inputs.cuda(opt.devices[0], async=True)

    if phase in ["Train"]:
        inputs_var = Variable(inputs, requires_grad=True)
        # logging.info("Switch to Train Mode")
        model.train()
    elif phase in ["Validate", "Test"]:
        inputs_var = Variable(inputs, volatile=True)
        # logging.info("Switch to Test Mode")
        model.eval()

    # forward
    if opt.cuda:
        if len(opt.devices) > 1:
            output = nn.parallel.data_parallel(model, inputs_var, opt.devices)
        else:
            output = model(inputs_var)
    else:
        output = model(inputs_var)
        print(output)

    # calculate loss for softmax
    target_vars = list()
    for index in range(len(target_softmax)):
        if opt.cuda:
            target_softmax[index] = target_softmax[index].cuda(opt.devices[0], async=True)
        target_vars.append(Variable(target_softmax[index]))
    loss_list = list()
    loss = Variable(torch.FloatTensor(1)).zero_()
    if opt.cuda:
        loss = loss.cuda(opt.devices[0])
    for index in range(len(target_softmax)):
        tmp1=target_vars[index]
        tmp2=output[index][0:opt.numctg]
        sub_loss = criterion_softmax(output[index][0:opt.numctg], target_vars[index])
        loss_list.append(sub_loss.data[0])
        loss += sub_loss
    #Calculate loss for sigmoid
    target_vars = list()
    for index in range(len(target_binary)):
        if opt.cuda:
            target_binary[index] = target_binary[index].cuda(opt.devices[0], async=True)
        target_vars.append(Variable(target_binary[index]))
    loss_list = list()
    loss = Variable(torch.FloatTensor(1)).zero_()
    if opt.cuda:
        loss = loss.cuda(opt.devices[0])
    for index in range(len(target_binary)):
        sub_loss = criterion_binary(output[index][opt.numctg:], target_vars[index])
        loss_list.append(sub_loss.data[0])
        loss += sub_loss
    return output, loss, loss_list


def forward_dataset(model, criterion_softmax, criterion_binary, data_loader, opt):
    sum_batch = 0
    accuracy = list()
    avg_loss = list()
    for i, data in enumerate(data_loader):
        if opt.mode == "Train":
            if random.random() > opt.validate_ratio:
                continue
        if opt.mode == "Test":
            logging.info("test %s/%s image" % (i, len(data_loader)))
        sum_batch += 1
        inputs, target_softmax,target_binary = data
        output, loss, loss_list = forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, "Validate")
        batch_accuracy = calc_accuracy(output, target_softmax,target_binary, opt.score_thres, opt.top_k)
        # accumulate accuracy
        if len(accuracy) == 0:
            accuracy = copy.deepcopy(batch_accuracy)
            for index, item in enumerate(batch_accuracy):
                for k, v in item.iteritems():
                    accuracy[index][k]["ratio"] = v["ratio"]
        else:
            for index, item in enumerate(batch_accuracy):
                for k, v in item.iteritems():
                    accuracy[index][k]["ratio"] += v["ratio"]
        # accumulate loss
        if len(avg_loss) == 0:
            avg_loss = copy.deepcopy(loss_list)
        else:
            for index, loss in enumerate(loss_list):
                avg_loss[index] += loss
    # average on batches
    for index, item in enumerate(accuracy):
        for k, v in item.iteritems():
            accuracy[index][k]["ratio"] /= float(sum_batch)
    for index in range(len(avg_loss)):
        avg_loss[index] /= float(sum_batch)
    return accuracy, avg_loss


def calc_accuracy(outputs, targets, score_thres, top_k=(1,)):
    max_k = max(top_k)
    accuracy = []
    thres_list = eval(score_thres)
    if isinstance(thres_list, float) or isinstance(thres_list, int):
        thres_list = [eval(score_thres)] * len(targets)

    for i in range(len(targets)):
        target = targets[i]
        output = outputs[i].data
        batch_size = output.size(0)
        curr_k = min(max_k, output.size(1))
        top_value, index = output.cpu().topk(curr_k, 1)
        index = index.t()
        top_value = top_value.t()
        correct = index.eq(target.cpu().view(1, -1).expand_as(index))
        mask = (top_value >= thres_list[i])
        correct = correct * mask
        # print "masked correct: ", correct
        res = defaultdict(dict)
        for k in top_k:
            k = min(k, output.size(1))
            correct_k = correct[:k].view(-1).float().sum(0)[0]
            res[k]["s"] = batch_size
            res[k]["r"] = correct_k
            res[k]["ratio"] = float(correct_k) / batch_size
        accuracy.append(res)
    return accuracy


def train(model, criterion_softmax, criterion_binary, train_set, val_set, opt):
    # define web visualizer using visdom
    #webvis = WebVisualizer(opt)

    # modify learning rate of last layer
    finetune_params = modify_last_layer_lr(model.named_parameters(),
                                           opt.lr, opt.lr_mult_w, opt.lr_mult_b)
    # define optimizer
    optimizer = optim.SGD(finetune_params,
                          opt.lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    # define laerning rate scheluer
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)


    # record forward and backward times
    train_batch_num = len(train_set)
    total_batch_iter = 0
    logging.info("####################Train Model###################")
    for epoch in range(opt.sum_epoch):
       # epoch_start_t = time.time()
        epoch_batch_iter = 0
        logging.info('Begin of epoch %d' % (epoch))
        for i, data in enumerate(train_set):
           # iter_start_t = time.time()
            # train
            inputs, target_softmax,target_binary = data
            output, loss, loss_list = forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, "Train")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

           # webvis.reset()
            epoch_batch_iter += 1
            total_batch_iter += 1




      #  logging.info('End of epoch %d / %d \t Time Taken: %d sec' %
                    # (epoch, opt.sum_epoch, time.time() - epoch_start_t))

        if epoch % opt.save_epoch_freq == 0:
            logging.info('saving the model at the end of epoch %d, iters %d' % (epoch + 1, total_batch_iter))
            save_model(model, opt, epoch + 1)

            # adjust learning rate
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        logging.info('learning rate = %.7f epoch = %d' % (lr, epoch))
    logging.info("--------Optimization Done--------")


def validate(model, criterion_softmax, criterion_binary, val_set, opt):
    return forward_dataset(model, criterion_softmax, criterion_binary, val_set, opt)


def test(model, criterion_softmax, criterion_binary, test_set, opt):
    logging.info("####################Test Model###################")
    test_accuracy, test_loss = forward_dataset(model, criterion_softmax, criterion_binary, test_set, opt)
    logging.info("data_dir:   " + opt.data_dir + "/TestSet/")
    logging.info("score_thres:" + str(opt.score_thres))
    for index, item in enumerate(test_accuracy):
        logging.info("Attribute %d:" % (index))
        for top_k, value in item.iteritems():
            logging.info("----Accuracy of Top%d: %f" % (top_k, value["ratio"]))
    logging.info("#################Finished Testing################")


def main():
    # parse options
    op = Options()
    opt = op.parse()

    # initialize train or test working dir
    trainer_dir = "trainer_" + opt.name
    opt.model_dir = os.path.join(opt.dir, trainer_dir, "Train")
    opt.data_dir = os.path.join(opt.dir, trainer_dir, "Data")
    opt.test_dir = os.path.join(opt.dir, trainer_dir, "Test")

    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)
    if opt.mode == "Train":
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        log_dir = opt.model_dir
        log_path = log_dir + "/train.log"
    if opt.mode == "Test":
        if not os.path.exists(opt.test_dir):
            os.makedirs(opt.test_dir)
        log_dir = opt.test_dir
        log_path = log_dir + "/test.log"

    # save options to disk
    util.opt2file(opt, log_dir + "/opt.txt")

    # log setting
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)

    # load train or test data
    ds = DeepFashionDataset(opt)
    num_data = len(ds)
    indices = list(range(num_data))
    split = int((opt.ratio[1] + opt.ratio[2]) * num_data)
    validation_Test_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_Test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    # validation Set
    split = int(round(0.5 * len(validation_Test_idx)))
    validation_idx = np.random.choice(validation_Test_idx, size=split, replace=False)
    validation_sampler = SubsetRandomSampler(validation_idx)
    # Test set
    test_idx = list(set(validation_Test_idx) - set(validation_idx))
    test_sampler = SubsetRandomSampler(test_idx)

    train_set = DataLoader(ds, batch_size=opt.batch_size, shuffle=False, sampler=train_sampler)
    val_set= DataLoader(ds, batch_size=opt.batch_size, shuffle=False, sampler=validation_sampler)
    test_set = DataLoader(ds, batch_size=opt.batch_size, shuffle=False, sampler=test_sampler)



    num_classes = [opt.numctg,opt.numattri] #temporary lets put the number of class []
    opt.class_num = len(num_classes)

    # load model
    model = load_model(opt, num_classes)

    # define loss function
    criterion_softmax = nn.CrossEntropyLoss(weight=opt.loss_weight)
    criterion_binary=torch.nn.BCELoss()


    # use cuda
    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion_softmax = criterion_softmax.cuda(opt.devices[0])
        criterion_binary= criterion_binary.cuda(opt.devices[0])
        cudnn.benchmark = True

    # Train model
    if opt.mode == "Train":
        train(model, criterion_softmax,criterion_binary, train_set, val_set, opt)
    # Test model
    elif opt.mode == "Test":
        test(model, criterion_softmax,criterion_binary, test_set, opt)


if __name__ == "__main__":
    main()