"""
Original Code is from https://github.com/shengliu66/SOP
"""

import argparse
import collections
import sys
import torch
import loader.sop_dataloader as module_data
import generic.loss as module_loss
import generic.sop_model as module_arch
import generic.sop_metric as module_metric
from generic.sop_parse_config import ConfigParser
from generic.sop_trainer import Trainer
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
    reparametrization_net = None#config.initialize('reparam_arch', module_arch)  

    # get function handles of loss and metrics
    # logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)

    config['train_loss']['args']['num_examp'] = num_examp

    train_loss = config.initialize('train_loss', module_loss)

    val_loss = config.initialize('val_loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = [{'params': [p for p in model.parameters() if  getattr(p, 'requires_grad', False)]}
                        ]
    reparam_params = [{'params': train_loss.u, 'lr': config['lr_u'], 'weight_decay': config['optimizer_overparametrization']['args']['weight_decay']},
                      {'params': train_loss.v, 'lr': config['lr_v'], 'weight_decay': config['optimizer_overparametrization']['args']['weight_decay']}
                     ]#, 'momentum': config['optimizer_overparametrization']['args']['momentum']}] 

    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    optimizer_overparametrization = config.initialize('optimizer_overparametrization', torch.optim, reparam_params)


    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    lr_scheduler_overparametrization = None

    test_log = open('../eval_results/%s/%s/r%.1f/%s' % (config['data_loader']['name'], config['arch']['type'], config['trainer']['percent'], config['trainer']['log']) + '.log', 'a')
    test_log.write('===========================================\n')
    test_log.write('Eval with SOP ..\n')
    test_log.flush()
    trainer = Trainer(model, reparametrization_net, train_loss, metrics, optimizer, optimizer_overparametrization,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      lr_scheduler_overparametrization = lr_scheduler_overparametrization,
                      val_criterion=val_loss,
                      test_log=test_log)

    trainer.train()
    # logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']
    test_log.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch SOP')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=int,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--lr_u', '--learning_rate_u'], type=float, target=('lr_u',)),
        CustomArgs(['--lr_v', '--learning_rate_v'], type=float, target=('lr_v',)),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--instance', '--instance'], type=bool, target=('trainer', 'instance')),
        CustomArgs(['--att', '--att'], type=bool, target=('trainer', 'att')),
        CustomArgs(['--noise_file', '--noise_file'], type=str, target=('trainer', 'noise_file')),
        CustomArgs(['--data_dir', '--data_dir'], type=str, target=('data_loader', 'args', 'data_dir')),
        CustomArgs(['--log', '--log'], type=str, target=('trainer', 'log')),
        CustomArgs(['--net', '--net'], type=str, target=('arch', 'type')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--key', '--comet_key'], type=str, target=('comet','api')),
        CustomArgs(['--offline', '--comet_offline'], type=str, target=('comet','offline')),
        CustomArgs(['--std', '--standard_deviation'], type=float, target=('reparam_arch','args','std')),
        CustomArgs(['--malpha', '--mixup_alpha'], type=float, target=('mixup','alpha')),
        CustomArgs(['--consist', '--ratio_consistency'], type=float, target=('train_loss','args','ratio_consistency')),
        CustomArgs(['--balance', '--ratio_balance'], type=float, target=('train_loss','args','ratio_balance'))
    ]
    config = ConfigParser.get_instance(args, options)

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    main(config)
