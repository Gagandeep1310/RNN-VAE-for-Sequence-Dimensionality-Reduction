import torch
import torch.nn as nn
import torch.optim as optim
from data import load_ptb
from model import RNNVAE
from tqdm import tqdm

def loss_function(logits, targets, mu, logvar, pad_idx, beta=1.0):
    recon_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

def train_model(epochs=5, latent_dim=32, embed_size=128, hidden_size=256, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader, vocab = load_ptb(batch_size=batch_size)
    pad_idx = vocab["<pad>"]

    model = RNNVAE(len(vocab), embed_size, hidden_size, latent_dim, pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(x)
            loss, recon, kl = loss_function(logits, y, mu, logvar, pad_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon
            total_kl += kl
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Recon: {total_recon/len(train_loader):.4f} | KL: {total_kl/len(train_loader):.4f}")

    torch.save(model.state_dict(), "rnn_vae.pt")
    print("Model saved.")

if __name__ == "__main__":
    train_model()

