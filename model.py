import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h = h[-1]  # final hidden state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, z):
        emb = self.embedding(x)
        h0 = self.latent_to_hidden(z).unsqueeze(0)  # init hidden
        out, _ = self.gru(emb, h0)
        logits = self.fc_out(out)
        return logits

class RNNVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_dim, pad_idx):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, latent_dim, pad_idx)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, latent_dim, pad_idx)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(x, z)
        return logits, mu, logvar

