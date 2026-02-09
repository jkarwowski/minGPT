"""
Trains a character-level language model.
"""

import argparse
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging
from mingpt.configs import CharGPTConfig, load_config

# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a character-level language model.")
    parser.add_argument("--config", default="configs/chargpt.yaml", help="Path to YAML config.")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. trainer.batch_size=128")
    return parser.parse_args()

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, config, data, vocab=None):
        self.config = config

        if vocab is None:
            chars = sorted(list(set(data)))
            data_size, vocab_size = len(data), len(chars)
            print('data has %d characters, %d unique.' % (data_size, vocab_size))

            self.stoi = { ch:i for i,ch in enumerate(chars) }
            self.itos = { i:ch for i,ch in enumerate(chars) }
            self.vocab_size = vocab_size
        else:
            self.stoi, self.itos = vocab
            self.vocab_size = len(self.stoi)
            data_size = len(data)
            print('data has %d characters, %d unique.' % (data_size, self.vocab_size))
            unknown = set(data) - set(self.stoi.keys())
            assert not unknown, f"validation data contains unseen characters: {sorted(unknown)[:10]}"
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    args = parse_args()
    config = load_config(args.config, CharGPTConfig, overrides=args.set)
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training and validation datasets
    text = open(config.data.input_path, 'r').read() # don't worry we won't run out of file handles
    split = int(config.data.train_split * len(text))
    train_text = text[:split]
    val_text = text[split:]
    chars = sorted(list(set(text)))
    vocab = ({ ch:i for i,ch in enumerate(chars) }, { i:ch for i,ch in enumerate(chars) })
    train_dataset = CharDataset(config.data, train_text, vocab=vocab)
    val_dataset = CharDataset(config.data, val_text, vocab=vocab)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(
        config.trainer,
        model,
        train_dataset,
        val_dataset=val_dataset,
        run_config=config.to_dict(),
    )

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
