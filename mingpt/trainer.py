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
        # auto batch size search
        C.auto_batch_size = False
        C.auto_batch_size_start = None
        C.auto_batch_size_factor = 2
        C.auto_batch_size_max = 4096
        # lr schedule
        C.lr_schedule = CN()
        C.lr_schedule.name = 'constant'
        C.lr_schedule.warmup_iters = 0
        C.lr_schedule.max_iters = 0
        C.lr_schedule.min_lr = 0.0
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

    def _batch_fits(self, batch_size):
        model, config = self.model, self.config
        pin_memory = self.device == 'cuda'
        loader = DataLoader(
            self.train_dataset,
            shuffle=False,
            pin_memory=pin_memory,
            batch_size=batch_size,
            num_workers=0,
        )
        try:
            batch = next(iter(loader))
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            model.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            return True
        except RuntimeError as exc:
            msg = str(exc).lower()
            if (
                'out of memory' in msg
                or 'cublas_status_alloc_failed' in msg
                or 'cuda error' in msg and 'memory' in msg
            ):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            raise

    def find_max_batch_size(self):
        config = self.config
        try:
            dataset_size = len(self.train_dataset)
        except TypeError:
            dataset_size = None

        max_bs = config.auto_batch_size_max
        if dataset_size is not None:
            max_bs = min(max_bs, dataset_size) if max_bs is not None else dataset_size
        if max_bs is None:
            max_bs = config.batch_size

        if max_bs < 1:
            raise ValueError("auto_batch_size_max must be >= 1")

        factor = max(2, int(config.auto_batch_size_factor or 2))
        start = config.auto_batch_size_start or config.batch_size or 1
        start = max(1, int(start))
        if max_bs is not None:
            start = min(start, max_bs)

        fits_cache = {}

        def fits(bs):
            if bs not in fits_cache:
                fits_cache[bs] = self._batch_fits(bs)
            return fits_cache[bs]

        bs = start
        if not fits(bs):
            while bs > 1 and not fits(bs):
                bs = max(1, bs // factor)
            if not fits(bs):
                raise RuntimeError("No viable batch size found (even 1 does not fit).")

        next_bs = min(max_bs, bs * factor)
        while next_bs <= max_bs and fits(next_bs):
            bs = next_bs
            next_bs = min(max_bs, bs * factor)
            if next_bs == bs:
                break

        low = bs + 1
        high = min(max_bs, next_bs - 1)
        while low <= high:
            mid = (low + high) // 2
            if fits(mid):
                bs = mid
                low = mid + 1
            else:
                high = mid - 1

        return bs

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

    def _get_lr(self, it):
        config = self.config
        schedule = getattr(config, 'lr_schedule', None)
        if schedule is None:
            return self.optimizer.param_groups[0]['lr']

        name = schedule.name
        if name == 'constant':
            return config.learning_rate

        warmup = schedule.warmup_iters
        max_iters = schedule.max_iters
        min_lr = schedule.min_lr

        if warmup > 0 and it < warmup:
            return config.learning_rate * (it + 1) / warmup

        if max_iters <= warmup:
            return config.learning_rate

        if it > max_iters:
            return min_lr

        decay_ratio = (it - warmup) / (max_iters - warmup)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)

        if name == 'linear':
            return min_lr + (1.0 - decay_ratio) * (config.learning_rate - min_lr)
        if name == 'cosine':
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (config.learning_rate - min_lr)
        if name == 'exp':
            if min_lr <= 0.0:
                raise ValueError("exp lr_schedule requires min_lr > 0")
            ratio = min_lr / config.learning_rate
            return config.learning_rate * (ratio ** decay_ratio)

        raise ValueError(f"unknown lr_schedule.name {name!r}")

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

        if config.auto_batch_size:
            max_bs = self.find_max_batch_size()
            config.batch_size = max_bs
            print(f"auto batch size set to {max_bs}")
            self.log_metrics({'train/auto_batch_size': int(max_bs)}, step=0)

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
                lr = self._get_lr(self.iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
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
