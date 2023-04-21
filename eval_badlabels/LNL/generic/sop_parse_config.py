import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
# from logger import setup_logging
import torch.cuda

from .utils import read_json, write_json


class ConfigParser:

    __instance = None

    def __new__(cls, args, options='', timestamp=True):
        raise NotImplementedError('Cannot initialize via Constructor')

    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    @classmethod
    def get_instance(cls, args=None, options='', timestamp=True):
        if not cls.__instance:
            if args is None:
                NotImplementedError('Cannot initialize without args')
            cls.__instance = cls.__internal_new__()
            cls.__instance.__init__(args, options)

        return cls.__instance

    def __init__(self, args, options='', timestamp=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()
        self.args = args

        if args.device:
            torch.cuda.set_device(args.device)
            # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is None:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.cfg_fname = Path(args.config)
            config = read_json(self.cfg_fname)
            self.resume = None
        else:
            self.resume = Path(args.resume)
            resume_cfg_fname = self.resume.parent / 'config.json'
            config = read_json(resume_cfg_fname)
            if args.config is not None:
                config.update(read_json(Path(args.config)))

        # load config file and apply custom cli options
        self._config = _update_config(config, options, args)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''


        if self.config['trainer']['asym']:
            exper_name = self.config['data_loader']['name'] + '_' + self.config['arch']['type'] + '_asym_' + str(int(self.config['trainer']['percent']*100))
        elif self.config['trainer']['instance']:
            exper_name = self.config['data_loader']['name'] + '_' + self.config['arch']['type'] + '_instance_' + str(int(self.config['trainer']['percent']*100))
        elif self.config['trainer']['att']:
            exper_name = self.config['data_loader']['name'] + '_' + self.config['arch']['type'] + '_att_' + str(int(self.config['trainer']['percent']*100))
        else:
            exper_name = self.config['data_loader']['name'] + '_' + self.config['arch']['type'] + '_sym_' + str(int(self.config['trainer']['percent']*100))
        self._save_dir = save_dir / 'models' / exper_name / timestamp
        self._log_dir = save_dir / 'log' / exper_name / timestamp

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        # setup_logging(self.log_dir)
        # self.log_levels = {
        #     0: logging.WARNING,
        #     1: logging.INFO,
        #     2: logging.DEBUG
        # }

    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the 
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    # def get_logger(self, name, verbosity=2):
    #     msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
    #                                                                                    self.log_levels.keys())
    #     assert verbosity in self.log_levels, msg_verbosity
    #     logger = logging.getLogger(name)
    #     logger.setLevel(self.log_levels[verbosity])
    #     return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def device(self):
        return self.args.device


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
            if 'target2' in opt._fields:
                _set_by_path(config, opt.target2, value)
            if 'target3' in opt._fields:
                _set_by_path(config, opt.target3, value)

    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
