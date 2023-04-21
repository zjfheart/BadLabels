import numpy as np
import torch
# from logger import CometWriter
from abc import abstractmethod
from numpy import inf
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from .utils import inf_loop
import sys
import torch.nn.functional as F


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, reparametrization_net, train_criterion, metrics, optimizer, optimizer_loss, config,
                 val_criterion):
        self.config = config
        # self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        if config['comet']['api'] is not None:
            self.writer = None
            # self.writer = CometWriter(
            #     self.logger,
            #     project_name=config['comet']['project_name'],
            #     experiment_name=config['name'],
            #     api_key=config['comet']['api'],
            #     log_dir=config.log_dir,
            #     offline=config['comet']['offline'])
        else:
            self.writer = None

        if self.writer is not None:
            self.writer.log_hyperparams(config.config)
            self.writer.log_code()

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'], config.device)
        self.model = model.to(self.device)

        if reparametrization_net is not None:
            self.reparametrization_net = reparametrization_net.to(self.device)
        else:
            self.reparametrization_net = None

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            if reparametrization_net is not None:
                self.reparametrization_net = torch.nn.DataParallel(reparametrization_net, device_ids=device_ids)

        self.train_criterion = train_criterion.to(self.device)

        self.val_criterion = val_criterion
        self.metrics = metrics

        self.optimizer = optimizer
        self.optimizer_loss = optimizer_loss

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epochs number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
            if epoch <= self.config['trainer']['warmup']:
                result = self._warmup_epoch(epoch)
            else:
                result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'test_metrics':
                    log.update({'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            # for key, value in log.items():
            #     self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    # self.logger.warning("Warning: Metric '{}' is not found. "
                    #                     "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    # self.logger.info("Validation performance didn\'t improve for {} epochs. "
                    #                  "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        if self.writer is not None:
            self.writer.finalize()

    def _prepare_device(self, n_gpu_use, gpu_id):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            # self.logger.warning("Warning: There\'s no GPU available on this machine,"
            #                     "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            # self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            #                     "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device(f'cuda:{gpu_id}' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        # torch.save(state, filename)
        # self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            # self.logger.info("Saving current best: model_best.pth at: {} ...".format(best_path))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        # self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, reparametrization_net, train_criterion, metrics, optimizer, optimizer_loss, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, lr_scheduler_overparametrization=None, len_epoch=None, val_criterion=None, test_log=None):
        super().__init__(model, reparametrization_net, train_criterion, metrics, optimizer, optimizer_loss, config, val_criterion)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader

        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_overparametrization = lr_scheduler_overparametrization
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []

        self.train_criterion = train_criterion

        self.new_best_val = False
        self.val_acc = 0
        self.test_val_acc = 0

        self.test_log = test_log

        

    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, label)
            if self.writer is not None:
                self.writer.add_scalar({'{}'.format(metric.__name__): acc_metrics[i]})
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        if self.reparametrization_net is not None:
            self.reparametrization_net.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        noise_level = 0


        with tqdm(self.data_loader) as progress:
            for batch_idx, (data, data2, label, indexs, _) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')


                data, label = data.to(self.device), label.long().to(self.device)

                target = torch.zeros(len(label), self.config['num_classes']).to(self.device).scatter_(1, label.view(-1,1), 1)
    

                if self.config['train_loss']['args']['ratio_consistency'] > 0:
                    data2 = data2.to(self.device)
                    data_all = torch.cat([data, data2]).cuda()
                else:
                    data_all = data
                    
                output = self.model(data_all)
                

                loss = self.train_criterion(indexs, output, target)

                self.optimizer_loss.zero_grad()
                self.optimizer.zero_grad()
                
                
                loss.backward()

                self.optimizer_loss.step()
                self.optimizer.step()


                if self.config['train_loss']['args']['ratio_consistency'] > 0:
                    output, _ = torch.chunk(output, 2)

                if self.writer is not None:

                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, epoch=epoch)           
                    self.writer.add_scalar({'loss': loss.item()})

                
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))

                if batch_idx == self.len_epoch:
                    break


        log = {
            'loss': total_loss / self.len_epoch,
            'noise level': noise_level/ self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler.get_lr()
        }


        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)



        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # if self.lr_scheduler_overparametrization is not None:
        #     self.lr_scheduler_overparametrization.step()

        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, label, indexs, _) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    if self.reparametrization_net is not None:
                        output, original_output = self.reparametrization_net(output, indexs)
                    loss = self.val_criterion(output, label)

                    if self.writer is not None:
                        self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, epoch=epoch, mode = 'valid')
                        self.writer.add_scalar({'loss': loss.item()})
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(config.cpu(), nrow=8, normalize=True))

        val_acc = (total_val_metrics / len(self.valid_data_loader)).tolist()[0]
        if val_acc > self.val_acc:
            self.val_acc = val_acc
            self.new_best_val = True
            if self.writer is not None:
                self.writer.add_scalar({'Best val acc': self.val_acc}, epoch = epoch)
        else:
            self.new_best_val = False

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch

        :return: A log that contains information about test

        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        results = np.zeros((len(self.test_data_loader.dataset), self.config['num_classes']), dtype=np.float32)
        tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, label,indexs,_) in enumerate(progress):
                    progress.set_description_str(f'Test epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    if self.reparametrization_net is not None:
                        output, original_output = self.reparametrization_net(output, indexs)
                    loss = self.val_criterion(output, label)
                    if self.writer is not None:
                        self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, epoch=epoch, mode = 'test')
                        self.writer.add_scalar({'loss': loss.item()})
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(config.cpu(), nrow=8, normalize=True))

                    results[indexs.cpu().detach().numpy().tolist()] = output.cpu().detach().numpy().tolist()
                    tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()

        # add histogram of model parameters to the tensorboard
        top_1_acc = (total_test_metrics / len(self.test_data_loader)).tolist()[0]
        if self.new_best_val:
            self.test_val_acc = top_1_acc
            if self.writer is not None:
                self.writer.add_scalar({'Test acc with best val': top_1_acc}, epoch = epoch)
        if self.writer is not None:
            self.writer.add_scalar({'Top-1': top_1_acc}, epoch = epoch)
            self.writer.add_scalar({'Top-5': (total_test_metrics / len(self.test_data_loader)).tolist()[1]}, epoch = epoch)

        self.test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, top_1_acc*100.))
        self.test_log.flush()

        return {
            'test_loss': total_test_loss / len(self.test_data_loader),
            'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
        }



    def _warmup_epoch(self, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.model.train()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()

        data_loader = self.data_loader#self.loader.run('warmup')


        with tqdm(data_loader) as progress:
            for batch_idx, (data, _, label, indexs , _) in enumerate(progress):
                progress.set_description_str(f'Warm up epoch {epoch}')

                data, label = data.to(self.device), label.long().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                if self.reparametrization_net is not None:
                    output, original_output = self.reparametrization_net(output, indexs)
                out_prob = torch.nn.functional.softmax(output).data.detach()

                loss = torch.nn.functional.cross_entropy(output, label)

                loss.backward() 
                self.optimizer.step()
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, epoch=epoch)
                    self.writer.add_scalar({'loss_record': loss.item()})
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))
                    # self.writer.add_image('input', make_grid(config.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
        if hasattr(self.data_loader, 'run'):
            self.data_loader.run()
        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler.get_lr()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
