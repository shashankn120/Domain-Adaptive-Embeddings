import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

# ──────────────────────────────────────────────
# 1. DOMAIN TRANSFORMS
# ──────────────────────────────────────────────

class DomainATransform:
    def __call__(self, img):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])(img)

class DomainBTransform:
    def __call__(self, img):
        return T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: torch.clamp(x * 3.0 - 1.0, -1, 1)),   # contrast shift
            T.Lambda(lambda x: T.functional.rotate(
                x.unsqueeze(0), angle=45, fill=0).squeeze(0)
            ),
            T.Lambda(lambda x: x + 0.15 * torch.randn_like(x)),        # Gaussian noise
            T.Normalize((0.5,), (0.5,))
        ])(img)


# ──────────────────────────────────────────────
# 2. CROSS-DOMAIN TRIPLET DATASET
# ──────────────────────────────────────────────

class CrossDomainTripletDataset(Dataset):
    """
    Each item: (anchor_A, positive_B_same_class, negative_B_diff_class)
    anchor    → Domain A (clean)
    positive  → Domain B (same label, augmented)
    negative  → Domain B (different label, augmented)
    """
    def __init__(self, root, train=True):
        base = torchvision.datasets.FashionMNIST(root=root, train=train, download=True)
        self.data   = base.data          # uint8 tensors
        self.labels = base.targets       # long tensors
        self.tf_a   = DomainATransform()
        self.tf_b   = DomainBTransform()

        # index: label → list of indices
        self.label_to_idx = {c: [] for c in range(10)}
        for i, lbl in enumerate(self.labels.tolist()):
            self.label_to_idx[lbl].append(i)

    def __len__(self):
        return len(self.data)

    def _pil(self, idx):
        from PIL import Image
        return Image.fromarray(self.data[idx].numpy(), mode='L')

    def __getitem__(self, idx):
        anchor_lbl = self.labels[idx].item()

        # positive: same class, random other index, Domain B
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.label_to_idx[anchor_lbl])

        # negative: different class, Domain B
        neg_lbl = random.choice([c for c in range(10) if c != anchor_lbl])
        neg_idx = random.choice(self.label_to_idx[neg_lbl])

        anchor   = self.tf_a(self._pil(idx))
        positive = self.tf_b(self._pil(pos_idx))
        negative = self.tf_b(self._pil(neg_idx))

        return anchor, positive, negative, anchor_lbl


class PairedEvalDataset(Dataset):
    """Returns (domain_A_img, domain_B_img, label) for validation."""
    def __init__(self, root):
        base = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)
        self.data   = base.data
        self.labels = base.targets
        self.tf_a   = DomainATransform()
        self.tf_b   = DomainBTransform()

    def __len__(self):
        return len(self.data)

    def _pil(self, idx):
        from PIL import Image
        return Image.fromarray(self.data[idx].numpy(), mode='L')

    def __getitem__(self, idx):
        img_a = self.tf_a(self._pil(idx))
        img_b = self.tf_b(self._pil(idx))
        return img_a, img_b, self.labels[idx]


# ──────────────────────────────────────────────
# 3. CNN FEATURE EXTRACTOR → 128-d embedding
# ──────────────────────────────────────────────

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),   # 14×14

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),   # 7×7

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),           # 1×1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.head(z)
        return nn.functional.normalize(z, dim=1)   # unit-sphere


# ──────────────────────────────────────────────
# 4. TRAINING LOOP (TripletMarginLoss)
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for anchor, positive, negative, _ in loader:
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        ea = model(anchor)
        ep = model(positive)
        en = model(negative)

        loss = criterion(ea, ep, en)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ──────────────────────────────────────────────
# 5. VALIDATION: EMBEDDINGS → PCA → TOP-1 ACC
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(model, eval_loader, device, use_tsne=False, n_components=32):
    model.eval()
    emb_a_list, emb_b_list, label_list = [], [], []

    for img_a, img_b, labels in eval_loader:
        ea = model(img_a.to(device)).cpu().numpy()
        eb = model(img_b.to(device)).cpu().numpy()
        emb_a_list.append(ea)
        emb_b_list.append(eb)
        label_list.append(labels.numpy())

    emb_a = np.concatenate(emb_a_list)   # (N, 128)
    emb_b = np.concatenate(emb_b_list)
    labels = np.concatenate(label_list)

    # Dimensionality reduction
    all_emb = np.concatenate([emb_a, emb_b], axis=0)
    if use_tsne:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=n_components, random_state=42)
    reduced = reducer.fit_transform(all_emb)

    N = len(emb_a)
    proj_a = reduced[:N]
    proj_b = reduced[N:]

    # Top-1 matching: for each Domain-A embedding find nearest Domain-B
    # using L2 in projected space
    # Build distance matrix (N, N)
    diff = proj_a[:, None, :] - proj_b[None, :, :]   # (N, N, d)
    dists = np.linalg.norm(diff, axis=-1)             # (N, N)
    nn_idx = np.argmin(dists, axis=1)                 # (N,)

    top1_acc = np.mean(labels[nn_idx] == labels)
    return top1_acc, proj_a, proj_b, labels


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

def main():
    # Config
    DATA_ROOT  = "./data"
    BATCH_SIZE = 128
    EPOCHS     = 10
    LR         = 3e-4
    MARGIN     = 0.3
    EMBED_DIM  = 128
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")

    # Datasets
    train_ds = CrossDomainTripletDataset(DATA_ROOT, train=True)
    eval_ds  = PairedEvalDataset(DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=256,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model     = EmbeddingNet(embed_dim=EMBED_DIM).to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        top1, _, _, _ = validate(model, eval_loader, DEVICE,
                                  use_tsne=False, n_components=32)

        flag = "★" if top1 > best_acc else ""
        if top1 > best_acc:
            best_acc = top1
            torch.save(model.state_dict(), "best_embedding_net.pt")

        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  "
              f"Top-1_A→B={top1:.4f} {flag}")

    print(f"\nBest Top-1 Cross-Domain Matching Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()