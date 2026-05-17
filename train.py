import random
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ── Paths ──────────────────────────────────────────────────────────────────
DESKTOP_DIR    = Path(r"C:\Users\arfao\Desktop\data_preprocessing_celeba")
IMAGE_DIR      = DESKTOP_DIR / "celeba_cgan"
LABEL_CSV      = DESKTOP_DIR / "celeba_cgan_labels.csv"
LOG_DIR        = DESKTOP_DIR / "logs_cgan__128_cp"
CHECKPOINT_DIR = DESKTOP_DIR / "checkpoints_cgan__128_cp"
SAMPLE_DIR     = DESKTOP_DIR / "samples_cgan__128_cp"

for d in [LOG_DIR, CHECKPOINT_DIR, SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Image ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128
IMG_CHANNELS = 3

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE       = 64           # bump from 48 -> 64 (bigger batch stabilizes BN)
EPOCHS           = 150
LR_G             = 1e-4         # was 2e-4 — lower G LR
LR_D             = 4e-4         # was 2e-4 — TTUR, D learns faster
R1_GAMMA         = 1.0          # was 0.05 — proper R1 strength
LABEL_SMOOTH     = 0.1
D_WARMUP_EPOCHS  = 0            # no longer needed with proper init & TTUR
EMA_BETA         = 0.999
IMAGES_PER_COMBO = 5000         # keep at 4000 (see discussion above)

# ── Architecture ───────────────────────────────────────────────────────────
Z_DIM     = 256
EMBED_DIM = 64

# ── Conditioning ───────────────────────────────────────────────────────────
ATTRIBUTES = ["Male", "Young", "Smiling"]
N_ATTRS    = len(ATTRIBUTES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train_logs.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Model components ───────────────────────────────────────────────────────

class ConditionalBatchNorm2d(nn.Module):
    """CBN where gamma/beta are predicted from label embedding.
    At init: gamma = 1, beta = 0 (i.e. plain BN behavior)."""
    def __init__(self, num_features: int, label_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.fc = nn.Linear(label_dim, num_features * 2)
        # FIX: zero-init both weight AND bias so residual gamma/beta start at 0
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma, beta = self.fc(label_emb).chunk(2, dim=1)
        gamma = 1.0 + gamma  # gamma starts at 1 because fc(...) starts at 0
        return out * gamma.view(-1, out.size(1), 1, 1) + beta.view(-1, out.size(1), 1, 1)


class UpsampleBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, label_dim: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.cbn  = ConditionalBatchNorm2d(out_c, label_dim)

    def forward(self, x: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        return F.relu(self.cbn(self.conv(self.up(x)), label_emb), inplace=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_dim  = EMBED_DIM * N_ATTRS
        self.embeddings = nn.ModuleList(
            [nn.Embedding(2, EMBED_DIM) for _ in range(N_ATTRS)]
        )
        self.fc  = nn.Linear(Z_DIM + self.label_dim, 512 * 4 * 4)
        self.up1 = UpsampleBlock(512, 512, self.label_dim)
        self.up2 = UpsampleBlock(512, 256, self.label_dim)
        self.up3 = UpsampleBlock(256, 128, self.label_dim)
        self.up4 = UpsampleBlock(128,  64, self.label_dim)
        self.up5 = UpsampleBlock( 64,  64, self.label_dim)
        self.out = nn.Conv2d(64, IMG_CHANNELS, kernel_size=3, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        # FIX: skip CBN's fc layers so their zero-init survives
        for name, m in self.named_modules():
            if "cbn.fc" in name:
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init on the tanh output so we don't saturate at init
        nn.init.xavier_normal_(self.out.weight, gain=0.5)
        nn.init.zeros_(self.out.bias)
        # Embeddings: small uniform
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = torch.cat(
            [self.embeddings[i](labels[:, i].long()) for i in range(N_ATTRS)], dim=1
        )
        x = self.fc(torch.cat([z, label_emb], dim=1)).view(-1, 512, 4, 4)
        x = self.up1(x, label_emb)
        x = self.up2(x, label_emb)
        x = self.up3(x, label_emb)
        x = self.up4(x, label_emb)
        x = self.up5(x, label_emb)
        return torch.tanh(self.out(x))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def sn_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_net = nn.Sequential(
            sn_block(IMG_CHANNELS,  64),   # 128 -> 64
            sn_block(64,           128),   # 64  -> 32
            sn_block(128,          256),   # 32  -> 16
            sn_block(256,          512),   # 16  -> 8
            sn_block(512,          512),   # 8   -> 4
        )
        conv_out_dim    = 512 * 4 * 4
        self.label_proj = nn.utils.spectral_norm(nn.Linear(N_ATTRS, 512))
        self.fc         = nn.utils.spectral_norm(nn.Linear(conv_out_dim + 512, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = self.conv_net(x).flatten(start_dim=1)
        features = F.leaky_relu(features, 0.2)  # activation before concat
        l_emb    = F.leaky_relu(self.label_proj(labels), 0.2)
        return self.fc(torch.cat([features, l_emb], dim=1))

# ── Losses / utilities ─────────────────────────────────────────────────────

def r1_penalty(real_logits: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        only_inputs=True,
    )[0]
    # Standard R1: 0.5 * ||grad||^2
    return 0.5 * grad.pow(2).flatten(start_dim=1).sum(dim=1).mean()


def discriminator_loss(d_real, d_fake, real_images, smooth=LABEL_SMOOTH):
    # Hinge loss with optional one-sided label smoothing
    loss_real = F.relu((1.0 - smooth) - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    penalty   = R1_GAMMA * r1_penalty(d_real, real_images)
    return loss_real + loss_fake + penalty, loss_real.item(), loss_fake.item(), penalty.item()


@torch.no_grad()
def update_ema(ema_model, model, beta):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(beta).add_(p.data, alpha=1.0 - beta)
    # Also copy BN running stats (important because EMA doesn't track them)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.data.copy_(b.data)

# ── Dataset ────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        stem = Path(self.df.at[idx, "image_id"]).stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = self.image_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"No image found for id '{stem}' in {self.image_dir}")

        with Image.open(img_path) as img:
            gray   = img.convert("RGB")
            tensor = self.transform(gray)

        labels = torch.tensor(self.df.loc[idx, ATTRIBUTES].values.astype(np.float32))
        return tensor, labels


def build_balanced_df(csv_path, images_per_combo):
    df = pd.read_csv(csv_path)
    parts = [grp.sample(n=min(images_per_combo, len(grp)), random_state=SEED)
             for _, grp in df.groupby(ATTRIBUTES)]
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Log per-combo counts
    counts = out.groupby(ATTRIBUTES).size()
    logger.info(f"Per-combo counts:\n{counts}")
    return out


def make_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 3 channels
    ])

# ── Training steps ─────────────────────────────────────────────────────────

def train_discriminator_step(G, D, optD, real, labels):
    B = real.size(0)
    optD.zero_grad(set_to_none=True)
    real = real.detach().clone().requires_grad_(True)
    with torch.no_grad():
        z    = torch.randn(B, Z_DIM, device=real.device)
        fake = G(z, labels)
    d_real  = D(real, labels)
    d_fake  = D(fake, labels)
    loss_d, l_real, l_fake, l_pen = discriminator_loss(d_real, d_fake, real)
    loss_d.backward()
    # Light clip only — mostly a safety net
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
    optD.step()
    return loss_d.item(), l_real, l_fake, l_pen, d_real.mean().item(), d_fake.mean().item()


def train_generator_step(G, D, optG, labels, device):
    B = labels.size(0)
    optG.zero_grad(set_to_none=True)
    z      = torch.randn(B, Z_DIM, device=device)
    fake   = G(z, labels)
    loss_g = -D(fake, labels).mean()
    loss_g.backward()
    # NO tight clipping on G — was killing learning. Use a loose safety net.
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)
    optG.step()
    return loss_g.item()

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Device: {DEVICE} | IMG_SIZE: {IMAGE_SIZE} | BATCH: {BATCH_SIZE} "
                f"| LR_G: {LR_G} | LR_D: {LR_D} | R1: {R1_GAMMA}")

    df      = build_balanced_df(LABEL_CSV, IMAGES_PER_COMBO)
    logger.info(f"Dataset: {len(df):,} samples")

    dataset = FaceDataset(df, IMAGE_DIR, make_transform())
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=True, drop_last=True)

    G     = Generator().to(DEVICE)
    D     = Discriminator().to(DEVICE)
    G_EMA = deepcopy(G).to(DEVICE)
    G_EMA.eval()
    for p in G_EMA.parameters():
        p.requires_grad_(False)

    logger.info(f"G params: {sum(p.numel() for p in G.parameters()):,}")
    logger.info(f"D params: {sum(p.numel() for p in D.parameters()):,}")

    # FIX: Adam betas (0.5, 0.999) — safer than (0.0, 0.999) without StyleGAN's full recipe
    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

    # Fixed samples for monitoring
    fixed_z      = torch.randn(16, Z_DIM, device=DEVICE)
    all_combos   = [[m, y, s] for m in [0, 1] for y in [0, 1] for s in [0, 1]]
    fixed_labels = torch.tensor(all_combos * 2, dtype=torch.float32, device=DEVICE)

    g_losses, d_losses = [], []

    for epoch in range(EPOCHS):
        G.train(); D.train()
        epoch_g, epoch_d = [], []
        epoch_dreal, epoch_dfake = [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", dynamic_ncols=True)
        for real, labels in pbar:
            real, labels = real.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # D step
            loss_d, l_real, l_fake, l_pen, d_real_m, d_fake_m = \
                train_discriminator_step(G, D, optD, real, labels)

            # G step
            loss_g = train_generator_step(G, D, optG, labels, DEVICE)

            update_ema(G_EMA, G, EMA_BETA)

            epoch_g.append(loss_g); epoch_d.append(loss_d)
            epoch_dreal.append(d_real_m); epoch_dfake.append(d_fake_m)
            pbar.set_postfix(
                G=f"{loss_g:.3f}",
                D=f"{loss_d:.3f}",
                Dr=f"{d_real_m:+.2f}",
                Df=f"{d_fake_m:+.2f}",
            )

        avg_g, avg_d = float(np.mean(epoch_g)), float(np.mean(epoch_d))
        avg_dr, avg_df = float(np.mean(epoch_dreal)), float(np.mean(epoch_dfake))
        g_losses.append(avg_g); d_losses.append(avg_d)

        logger.info(f"Epoch {epoch+1:3d}/{EPOCHS} | G: {avg_g:.4f} | D: {avg_d:.4f} "
                    f"| D(real): {avg_dr:+.3f} | D(fake): {avg_df:+.3f}")

        # Samples every 2 epochs early, then every 5
        sample_every = 2 if epoch < 20 else 5
        if (epoch + 1) % sample_every == 0:
            G_EMA.eval()
            with torch.no_grad():
                samples = G_EMA(fixed_z, fixed_labels)
                vutils.save_image(samples, SAMPLE_DIR / f"epoch_{epoch+1:04d}.png",
                  normalize=True, value_range=(-1, 1), nrow=8)

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "g": G.state_dict(),
                "d": D.state_dict(),
                "g_ema": G_EMA.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
            }, CHECKPOINT_DIR / f"checkpoint_{epoch+1}.pth")

    # Final loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(LOG_DIR / "losses.png", dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()