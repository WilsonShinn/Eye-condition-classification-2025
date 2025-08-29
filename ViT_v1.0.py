"""
代码说明:
本代码使用ViT，属于模型初探版本，ViT的每一个步骤都在代码中拆解实现
本代码与ViT结构图相配套
在我们20k级的数据集上，表现为acc=0.95-0.97左右

"""

import os, time, csv, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt


DATA_ROOT = "/root/small_data_split_new/small_data_split"
IMG_SIZE  = 224
BATCH     = 64
EPOCHS    = 50 # 通过看图像，50轮有点多，20轮应该差不多
LR        = 5e-4
WEIGHT_DECAY = 1e-4
SEED      = 42

LOG_CSV   = "ViT_metrics.csv"
FIG_PNG   = "ViT_loss_curve.png"
CKPT_BEST = "vit_best_macro_f1.pt"
CSV_3WAY  = "vit_pred_test_3way.csv"

ANCHOR = "/small_data_split/"  # 用于csv输出的截断路径

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| name:", torch.cuda.get_device_name(0) if device.type=="cuda" else "-")

# Dataset
class EyesDataset(Dataset):
    exts = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
    def __init__(self, root_split, is_train, return_path: bool=False):
        self.return_path = return_path
        self.samples = []
        for cls_name, label in [("positive",0),("negative",1)]:
            d = os.path.join(root_split, cls_name)
            if not os.path.isdir(d): raise FileNotFoundError(d)
            for fn in os.listdir(d):
                fp = os.path.join(d, fn)
                if os.path.isfile(fp) and fn.lower().endswith(self.exts):
                    self.samples.append((fp, label))

        if is_train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), value='random'),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.samples)

    def _trim_path(self, fp: str) -> str:
        if ANCHOR and (ANCHOR in fp): return fp[fp.index(ANCHOR):]
        return os.path.basename(fp)

    def __getitem__(self, idx):
        fp, y = self.samples[idx]
        im = Image.open(fp).convert("RGB")
        x = self.tf(im)
        if not self.return_path: return x, y
        return x, y, self._trim_path(fp)

def build_loaders():
    train_ds = EyesDataset(os.path.join(DATA_ROOT,"train"), is_train=True)
    val_ds   = EyesDataset(os.path.join(DATA_ROOT,"val"),   is_train=False)
    test_ds  = EyesDataset(os.path.join(DATA_ROOT,"test"),  is_train=False)
    common = dict(batch_size=BATCH, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    return (
        DataLoader(train_ds, shuffle=True,  **common),
        DataLoader(val_ds,   shuffle=False, **common),
        DataLoader(test_ds,  shuffle=False, **common),
    )

# ViT (自实现)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=3, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=3, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTClassifier(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=192, layers=8,
                 heads=3, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.1, drop=0.1,
                 output_dim=1, in_chans=3):
        super().__init__()
        self.patch_embed = PatchEmbed(input_resolution, patch_size, in_chans, width)
        num_patches = self.patch_embed.num_patches
        scale = width ** -0.5
        self.cls_token = nn.Parameter(scale * torch.randn(1, 1, width))
        self.pos_embed = nn.Parameter(scale * torch.randn(1, 1 + num_patches, width))
        self.pos_drop  = nn.Dropout(drop)
        self.blocks = nn.ModuleList([EncoderBlock(width, heads, mlp_ratio, attn_drop, proj_drop, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, output_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)
    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, 1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks: x = blk(x)
        x = self.ln(x)[:, 0]
        return self.head(x).squeeze(1)

# Eval
@torch.no_grad()
def evaluate(model, loader, device, criterion=None, thr=0.0, title=""):
    model.eval()
    logits_all, y_all = [], []
    loss_sum, cnt = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        if criterion is not None:
            loss_sum += criterion(logits, yb.float()).item() * xb.size(0); cnt += xb.size(0)
        logits_all.append(logits.cpu()); y_all.append(yb.cpu())
    logits = torch.cat(logits_all).numpy(); y = torch.cat(y_all).numpy()
    y_pred = (logits >= thr).astype(np.int64)
    acc = accuracy_score(y, y_pred)
    cm  = confusion_matrix(y, y_pred, labels=[0,1])
    print(f"{title}ACC={acc:.4f}\n{title}Confusion [open=0, closed=1]:\n{cm}")
    print(classification_report(y, y_pred, target_names=["open(0)","closed(1)"]))
    avg_loss = (loss_sum / max(1,cnt)) if criterion is not None else None
    return acc, cm, avg_loss

@torch.no_grad()
def find_best_threshold_macro_f1(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        all_logits.append(model(xb).cpu()); all_y.append(yb)
    logits = torch.cat(all_logits).numpy(); y = torch.cat(all_y).numpy()
    thrs = np.linspace(-2.0, 2.0, 81)
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in thrs:
        yp = (logits >= t).astype(np.int64)
        f1_o = f1_score(y, yp, pos_label=0); f1_c = f1_score(y, yp, pos_label=1)
        macro = 0.5 * (f1_o + f1_c)
        if macro > best[0]: best = (macro, t, f1_o, f1_c)
    return float(best[1]), float(best[0]), float(best[2]), float(best[3])

# 导出预测的CSV，分成三类：open/closed/uncertain，可用于后续分析
@torch.no_grad()
def export_predictions(model, device, csv_path=CSV_3WAY, thr_low=None, thr_high=None):
    export_ds = EyesDataset(os.path.join(DATA_ROOT, "test"), is_train=False, return_path=True)
    export_ld = DataLoader(export_ds, batch_size=BATCH, shuffle=False,
                           num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    if not os.path.isfile(CKPT_BEST): raise FileNotFoundError(CKPT_BEST)
    ckpt = torch.load(CKPT_BEST, map_location=device)
    best_thr = float(ckpt.get("thr", 0.0))
    if thr_low  is None: thr_low  = best_thr - 0.5
    if thr_high is None: thr_high = best_thr + 0.5

    model.eval()
    rows = [("filepath", "true_label", "pred_label", "logit", "prob")]
    for xb, yb, rels in export_ld:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).cpu().numpy()
        probs  = torch.sigmoid(torch.from_numpy(logits)).numpy()
        yb     = yb.numpy()
        for i in range(len(yb)):
            logit, prob = float(logits[i]), float(probs[i])
            if   logit >= thr_high: pred = "closed"
            elif logit <= thr_low : pred = "open"
            else                  : pred = "uncertain"
            rows.append((rels[i], int(yb[i]), pred, logit, prob))
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[EXPORT] saved: {csv_path} | thr_low={thr_low:.3f}, thr_high={thr_high:.3f} (best_thr={best_thr:.3f})")

# main
def main():
    train_loader, val_loader, test_loader = build_loaders()
    model = ViTClassifier(
        input_resolution=224, patch_size=16, width=192, layers=8, heads=3, output_dim=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda" if device.type=="cuda" else "cpu")
    scheduler = OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS,
                           steps_per_epoch=len(train_loader),
                           pct_start=0.1, div_factor=10.0, final_div_factor=100.0)

    best_macro_f1, best_thr = -1.0, 0.0
    train_loss_steps, val_loss_epochs = [], []

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_step","train_loss","val_loss","val_acc",
                                "val_best_thr","val_macro_f1","val_f1_open","val_f1_closed"])

    global_step = 0
    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}", ncols=110)
        seen, t0 = 0, time.time()
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
                logits = model(xb); loss = criterion(logits, yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()

            global_step += 1
            train_loss_steps.append(loss.item())
            seen += xb.size(0); ips = seen / max(1e-6, time.time() - t0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", ips=f"{ips:.0f} img/s")

       # 每跑完一个epoch，跑一下验证集
        print("[VAL]")
        val_acc, _, val_loss = evaluate(model, val_loader, device, criterion, thr=0.0, title="VAL ")
        cur_thr, cur_macro_f1, f1_open, f1_closed = find_best_threshold_macro_f1(model, val_loader, device)
        print(f"[VAL] best_thr={cur_thr:.3f}  macro_F1={cur_macro_f1:.4f}")
        val_loss_epochs.append(val_loss if val_loss is not None else np.nan)
        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([ep, global_step, train_loss_steps[-1], val_loss,
                                    val_acc, cur_thr, cur_macro_f1, f1_open, f1_closed])

        if cur_macro_f1 > best_macro_f1:
            best_macro_f1, best_thr = cur_macro_f1, cur_thr
            torch.save({"model": model.state_dict(), "thr": best_thr}, CKPT_BEST)
            print(f"[CKPT] saved {CKPT_BEST}")

    # 跑完所有epoch后，加载最优权重，跑测试集
    print("[TEST] (loading best checkpoint)")
    if os.path.isfile(CKPT_BEST):
        ckpt = torch.load(CKPT_BEST, map_location=device)
        model.load_state_dict(ckpt["model"]); best_thr = ckpt["thr"]
    evaluate(model, test_loader, device, criterion=None, thr=best_thr, title="TEST ")

    # 导出CSV
    export_predictions(model, device, csv_path=CSV_3WAY)

    # 输出loss曲线
    plt.figure()
    def moving_avg(x, k=50):
        if len(x) < k: return x
        y = np.convolve(np.array(x, float), np.ones(k)/k, mode="valid")
        return np.concatenate([np.full(len(x)-len(y), y[0]), y])
    tl = moving_avg(train_loss_steps, 50)
    plt.plot(tl, label="train_loss (smoothed)")
    if len(val_loss_epochs) > 0:
        xs = np.linspace(0, len(tl), num=len(val_loss_epochs))
        plt.plot(xs, val_loss_epochs, marker="o", label="val_loss")
    plt.xlabel("training steps"); plt.ylabel("loss"); plt.title("Loss Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG_PNG, dpi=150)
    print(f"saved: {FIG_PNG}, {LOG_CSV}, {CSV_3WAY}")

if __name__ == "__main__":
    main()