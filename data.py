import torch
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

def load_ptb(batch_size=32, seq_len=30, min_freq=5):
    train_iter = PennTreebank(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"], min_freq=min_freq)
    vocab.set_default_index(vocab["<unk>"])

    def encode(text):
        return torch.tensor(vocab(tokenizer(text)), dtype=torch.long)

    def collate_batch(batch):
        encoded = [encode(x)[:seq_len] for x in batch]
        padded = pad_sequence(encoded, batch_first=True, padding_value=vocab["<pad>"])
        return padded[:, :-1], padded[:, 1:]  # input, target shifted

    train_iter, valid_iter, test_iter = PennTreebank(split=('train', 'valid', 'test'))
    train_loader = torch.utils.data.DataLoader(list(train_iter), batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(list(valid_iter), batch_size=batch_size, collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(list(test_iter), batch_size=batch_size, collate_fn=collate_batch)

    return train_loader, valid_loader, test_loader, vocab

