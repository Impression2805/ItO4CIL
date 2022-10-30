import sys
import os
import csv
import logging
import copy
import torch
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import numpy as np
import pandas as pd


def train(args):
    seed_list = copy.deepcopy(args.seed)
    device = copy.deepcopy(args.device)

    for seed in seed_list:
        args.seed = seed
        args.device = device
        _train(args)


def _train(args):
    logfilename = 'log/{}_{}_{}_{}_{}'.format(args.model_name, args.convnet_type,
                                                  args.dataset, args.init_cls, args.increment)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    _set_random()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    args.file_name = args.dataset + '_' + args.model_name + '_' + str(args.increment) + '_replaydata_' + str(args.memory_per_class)
    if not os.path.isdir(args.save_path + args.file_name + '/'):
        os.makedirs(args.save_path + args.file_name + '/')
    _set_device(args)
    data_manager = DataManager(args.dataset, args.shuffle, args.seed, args.init_cls, args.increment)
    model = factory.get_model(args.model_name, args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    average_acc_cnn = 0
    average_acc_nme = 0
    acc_all_cnn = []
    acc_all_nme = []
    acc_new_task = []

    acc_class = []
    acc_class_nme = []
    name = []

    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(0)
            nme_curve['top5'].append(0)

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))

        # average_acc_cnn = np.around(sum(cnn_curve['top1']) / (task + 1), decimals=2)
        # if nme_accy is not None:
        #     average_acc_nme = np.around(sum(nme_curve['top1']) / (task + 1), decimals=2)


        ########################## test for each class #################################
        # if task == 0:
        #     classes = [0, args.init_cls]
        # else:
        #     classes = [0, args.init_cls + task * args.increment]
        #
        # test_dset = data_manager.get_dataset(np.arange(classes[0], classes[1]), source='test', mode='test')
        # test_loader = DataLoader(test_dset, batch_size=100, shuffle=False, num_workers=8)

        # y_pred, y_true = model._eval_cnn(test_loader)
        # acc_old = model._evaluate_fair(y_pred, y_true)
        # acc_class.append(acc_old)

        # y_pred_nme, y_true_nme = model._eval_nme(test_loader, model._class_means)
        # acc_old_nme = model._evaluate_fair(y_pred_nme, y_true_nme)
        # acc_class_nme.append(acc_old_nme)

        # _name = str(task) + '_stage'
        # name.append(_name)

        ########################### test for up2now task ###############################
        acc_up2now_cnn = []
        acc_up2now_nme = []
        for i in range(task + 1):
            if i == 0:
                classes = [0, args.init_cls]
            else:
                classes = [args.init_cls + (i - 1) * args.increment, args.init_cls + i * args.increment]

            test_dset = data_manager.get_dataset(np.arange(classes[0], classes[1]), source='test', mode='test')
            test_loader = DataLoader(test_dset, batch_size=100, shuffle=False, num_workers=8)

            y_pred, y_true = model._eval_cnn(test_loader)
            cnn_accy = model._evaluate(y_pred, y_true)
            acc_up2now_cnn.append(cnn_accy['top1'])

            if nme_accy is not None:
                y_pred, y_true = model._eval_nme(test_loader, model._class_means)
                nme_accy = model._evaluate(y_pred, y_true)
                acc_up2now_nme.append(nme_accy['top1'])
            else:
                nme_accy = None
                acc_up2now_nme.append(0)

            if i == task:
                acc_new_task.append(cnn_accy['top1'])

        if task < data_manager.nb_tasks - 1:
            acc_up2now_cnn.extend((data_manager.nb_tasks - 1 - task) * [0])
            acc_up2now_nme.extend((data_manager.nb_tasks - 1 - task) * [0])
        acc_all_cnn.append(acc_up2now_cnn)
        acc_all_nme.append(acc_up2now_nme)


    ############################ accuracy #########################
    # acc_cnn = cnn_curve['top1']
    # acc_nme = nme_curve['top1']
    # acc = [acc_cnn, acc_nme]
    # name = ['cnn_curve', 'nme_curve']
    # df = pd.DataFrame()
    # for k in range(len(name)):
    #     df = pd.concat([df, pd.DataFrame({name[k]: acc[k]})], axis=1)
    # path = args.save_path + args.file_name + '/accResult.csv'
    # df.to_csv(path, encoding='gbk', index=False)


    a = np.array(acc_all_cnn)
    result_cnn = []
    for i in range(data_manager.nb_tasks):
        if i == 0:
            result_cnn.append(0)
        else:
            res = 0
            for j in range(i + 1):
                res += (np.max(a[:, j]) - a[i][j])
            res = res / i
            result_cnn.append(np.around(res, decimals=2))
    print('Forgetting result of cnn:')
    print(result_cnn)

    a = np.array(acc_all_nme)
    result_nme = []
    for i in range(data_manager.nb_tasks):
        if i == 0:
            result_nme.append(0)
        else:
            res = 0
            for j in range(i + 1):
                res += (np.max(a[:, j]) - a[i][j])
            res = res / i
            result_nme.append(np.around(res, decimals=2))
    print('Forgetting result of nme:')
    print(result_nme)


    ############################ ACC & forgetting #########################
    acc_cnn = cnn_curve['top1']
    acc_nme = nme_curve['top1']
    data_saved = [acc_cnn, acc_nme, result_cnn, result_nme]
    name = ['cnn_curve', 'nme_curve', 'cnn_forget', 'nme_forget']
    df = pd.DataFrame()
    for k in range(len(name)):
        df = pd.concat([df, pd.DataFrame({name[k]: data_saved[k]})], axis=1)
    path = args.save_path + args.file_name + '/accforgetResult.csv'
    df.to_csv(path, encoding='gbk', index=False)


def _set_device(args):
    device_type = args.device
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args.device = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
