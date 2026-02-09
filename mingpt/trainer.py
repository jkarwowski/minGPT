"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        # evaluation parameters
        C.eval_interval = 500
        C.eval_batches = None
        C.eval_batch_size = None
        # wandb logging
        C.wandb = CN()
        C.wandb.enabled = True
        C.wandb.project = 'codex-mingpt2'
        C.wandb.entity = None
        C.wandb.name = None
        C.wandb.tags = None
        C.wandb.group = None
        C.wandb.notes = None
        C.wandb.mode = None
        return C

    def __init__(self, config, model, train_dataset, val_dataset=None, run_config=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.grad_norm = 0.0
        self.tokens_per_sec = 0.0

        # optional wandb logging
        self.wandb = None
        self.wandb_run = None
        self._wandb_config = run_config
        self._setup_wandb()

    def _setup_wandb(self):
        wandb_cfg = getattr(self.config, 'wandb', None)
        if not wandb_cfg or not wandb_cfg.enabled:
            return
        try:
            import wandb
        except Exception as exc:
            print(f"wandb disabled because import failed: {exc}")
            return

        init_kwargs = {
            'project': wandb_cfg.project or 'codex-mingpt2',
            'config': self._wandb_config or self.config.to_dict(),
        }
        if wandb_cfg.entity is not None:
            init_kwargs['entity'] = wandb_cfg.entity
        if wandb_cfg.name is not None:
            init_kwargs['name'] = wandb_cfg.name
        if wandb_cfg.tags is not None:
            init_kwargs['tags'] = wandb_cfg.tags
        if wandb_cfg.group is not None:
            init_kwargs['group'] = wandb_cfg.group
        if wandb_cfg.notes is not None:
            init_kwargs['notes'] = wandb_cfg.notes
        if wandb_cfg.mode is not None:
            init_kwargs['mode'] = wandb_cfg.mode

        self.wandb = wandb
        try:
            self.wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            print(f"wandb disabled because init failed: {exc}")
            self.wandb = None
            self.wandb_run = None

    def log_metrics(self, metrics, step=None):
        if self.wandb_run is None:
            return
        if step is None:
            step = self.iter_num
        self.wandb.log(metrics, step=step)

    def evaluate(self):
        if self.val_dataset is None:
            return None

        model, config = self.model, self.config
        was_training = model.training
        model.eval()

        eval_batch_size = config.eval_batch_size or config.batch_size
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=eval_batch_size,
            num_workers=config.num_workers,
        )

        losses = []
        with torch.no_grad():
            for b, batch in enumerate(val_loader):
                if config.eval_batches is not None and b >= config.eval_batches:
                    break
                batch = [t.to(self.device) for t in batch]
                x, y = batch
                _, loss = model(x, y)
                losses.append(loss.item())

        if was_training:
            model.train()

        if not losses:
            avg_loss = float('nan')
        else:
            avg_loss = float(sum(losses) / len(losses))

        if math.isfinite(avg_loss):
            try:
                ppl = float(math.exp(avg_loss))
            except OverflowError:
                ppl = float('inf')
        else:
            ppl = float('inf')

        return {
            'val/loss': avg_loss,
            'val/ppl': ppl,
        }

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        try:
            while True:

                # fetch the next batch (x, y) and re-init iterator if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                batch = [t.to(self.device) for t in batch]
                x, y = batch

                # forward the model
                logits, self.loss = model(x, y)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.grad_norm = float(grad_norm)
                self.optimizer.step()

                self.trigger_callbacks('on_batch_end')
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow
                if self.iter_dt > 0:
                    self.tokens_per_sec = float(x.numel()) / self.iter_dt
                else:
                    self.tokens_per_sec = 0.0

                # per-step logging
                lr = self.optimizer.param_groups[0]['lr']
                self.log_metrics({
                    'train/loss': float(self.loss.item()),
                    'train/lr': float(lr),
                    'train/grad_norm': self.grad_norm,
                    'train/iter_dt': float(self.iter_dt),
                    'train/tokens_per_sec': float(self.tokens_per_sec),
                    'train/batch_size': int(x.size(0)),
                    'train/seq_len': int(x.size(1)),
                }, step=self.iter_num)

                if config.eval_interval is not None and self.val_dataset is not None:
                    if self.iter_num % config.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        if eval_metrics is not None:
                            self.log_metrics(eval_metrics, step=self.iter_num)

                # termination conditions
                if config.max_iters is not None and self.iter_num >= config.max_iters:
                    break
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()
