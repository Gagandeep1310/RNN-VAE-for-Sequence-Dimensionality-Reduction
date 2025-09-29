import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

def visualize_latent(model, data_loader, vocab, n_samples=500):
    model.eval()
    device = next(model.parameters()).device
    latents, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            mu, logvar = model.encoder(x)
            latents.append(mu.cpu())
            labels.extend([vocab.lookup_tokens(seq.tolist()) for seq in x])
            if len(latents) * x.size(0) >= n_samples:
                break
    latents = torch.cat(latents, dim=0)[:n_samples]
    tsne = TSNE(n_components=2)
    z2d = tsne.fit_transform(latents)
    plt.scatter(z2d[:,0], z2d[:,1], alpha=0.6)
    plt.title("Latent Space (t-SNE)")
    plt.show()

