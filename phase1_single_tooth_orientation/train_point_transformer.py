import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import os

from dataset import make_loader, mean_euclidean_error
from point_transformer_model import LandmarkPointTransformer

# ---------------- 超参数 ----------------
NUM_LANDMARKS = 5
BATCH_SIZE = 32
EPOCHS = 200
LR = 3e-4  # 初始学习率调低，防止 Transformer 初期的 QKV 崩盘


def landmark_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, gt, reduction='none')
    loss = loss.sum(dim=-1)

    valid_sum = mask.sum()
    if valid_sum == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return (loss * mask).sum() / valid_sum


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ⚠️ 请确保这里的路径指向你新生成的包含法向量的点云文件夹
    TRAIN_DIR = r"F:\NDCS_3DS_data\Single_Teeth_PC_6D\train"
    TEST_DIR = r"F:\NDCS_3DS_data\Single_Teeth_PC_6D\test"

    train_loader = make_loader(TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True, augment=True, num_workers=4)
    val_loader = make_loader(TEST_DIR, batch_size=BATCH_SIZE, shuffle=False, augment=False, num_workers=4)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = LandmarkPointTransformer(num_landmarks=NUM_LANDMARKS).to(device)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    scaler = GradScaler('cuda')

    best_val_err = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            pts = batch["points"].to(device)
            lm = batch["landmarks"].to(device)
            mask = batch["lm_mask"].to(device)
            tid = batch["tooth_id"].to(device)

            opt.zero_grad()
            # 核心防御机制：使用 bfloat16 防止 NaN
            with autocast('cuda', dtype=torch.bfloat16):
                pred = model(pts, tid)
                loss = landmark_loss(pred, lm, mask)

            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()

        # ---------------- 验证循环 ----------------
        model.eval()
        val_loss_total = 0.0
        val_err = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                lm = batch["landmarks"].to(device)
                mask = batch["lm_mask"].to(device)
                tid = batch["tooth_id"].to(device)

                with autocast('cuda', dtype=torch.bfloat16):
                    pred = model(pts, tid)
                    v_loss = landmark_loss(pred, lm, mask)

                val_loss_total += v_loss.item()
                err = mean_euclidean_error(pred, lm, mask)

                batch_size_actual = pts.size(0)
                val_err += err * batch_size_actual
                n_val += batch_size_actual

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_err = val_err / max(n_val, 1)

        sched.step(avg_val_err)
        current_lr = opt.param_groups[0]['lr']

        if avg_val_err < best_val_err:
            best_val_err = avg_val_err
            torch.save(model.state_dict(), "best_single_tooth_model_6d.pt")
            print(
                f"Epoch {epoch:03d} | LR: {current_lr:.1e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Err: {avg_val_err:.4f} mm  <-- New Best!")
        else:
            print(
                f"Epoch {epoch:03d} | LR: {current_lr:.1e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Err: {avg_val_err:.4f} mm")


if __name__ == "__main__":
    train()