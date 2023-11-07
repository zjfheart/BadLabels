"""
Original Code is from https://github.com/shengliu66/ELR
"""


import argparse
import collections
import sys
import requests
import socket
import torch
import loader.elr_dataloader as module_data
import generic.loss as module_loss
import generic.sop_metric as module_metric
import generic.sop_model as module_arch
from generic.sop_parse_config import ConfigParser
from generic.elr_trainer import Trainer
from collections import OrderedDict
import random



# def log_params(conf: OrderedDict, parent_key: str = None):
#     for key, value in conf.items():
#         if parent_key is not None:
#             combined_key = f'{parent_key}-{key}'
#         else:
#             combined_key = key
#
#         if not isinstance(value, OrderedDict):
#             mlflow.log_param(combined_key, value)
#         else:
#             log_params(value, combined_key)


def main(config: ConfigParser):

    # logger = config.get_logger('train')

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=config['data_loader']['args']['validation_split'],
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'] 
    )


    valid_data_loader = data_loader.split_validation()

    # test_data_loader = None

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()


    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    # logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)

    train_loss = getattr(module_loss, config['train_loss']['type'])(num_examp=num_examp, num_classes=config['num_classes'],
                                                            beta=config['train_loss']['args']['beta'])

    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    test_log = open('../eval_results/%s/%s/r%.1f/%s' % (config['data_loader']['name'], config['arch']['type'], config['trainer']['percent'], config['trainer']['log']) + '.log', 'a')
    test_log.write('===========================================\n')
    test_log.write('Eval with ELR ..\n')
    test_log.flush()
    trainer = Trainer(model, train_loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      val_criterion=val_loss,
                      test_log=test_log)

    trainer.train()
    # logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']
    test_log.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--lamb', '--lamb'], type=float, target=('train_loss', 'args', 'lambda')),
        CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--idn', '--idn'], type=bool, target=('trainer', 'idn')),
        CustomArgs(['--bad', '--bad'], type=bool, target=('trainer', 'bad')),
        CustomArgs(['--noise_file', '--noise_file'], type=str, target=('trainer', 'noise_file')),
        CustomArgs(['--data_dir', '--data_dir'], type=str, target=('data_loader', 'args', 'data_dir')),
        CustomArgs(['--net', '--net'], type=str, target=('arch', 'type')),
        CustomArgs(['--log', '--log'], type=str, target=('trainer', 'log'))
    ]
    config = ConfigParser.get_instance(args, options)

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    main(config)
