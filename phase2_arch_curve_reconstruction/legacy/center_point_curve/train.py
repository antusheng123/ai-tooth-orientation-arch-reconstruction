import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

# 从我们拆分的模块中导入
from dataset import load_real_data, DentalArchDataset, SEQ_LEN
from model import MaskedArchRegressor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. 加载数据
    # ==========================================
    # 之前是直接指向 \upper，现在指向父目录
    SEG_ROOT_BASE = r"F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train"
    KPT_ROOT_BASE = r"F:\NDCS_3DS_data\3DTeethLand_landmarks_train"

    print("Loading data...")
    all_arch_data, _ = load_real_data(SEG_ROOT_BASE, KPT_ROOT_BASE)
    print(f"Total upper arches loaded: {len(all_arch_data)}")

    if len(all_arch_data) == 0: return

    # 划分训练/测试集
    random.shuffle(all_arch_data)
    split_idx = int(len(all_arch_data) * 0.8)
    train_data = all_arch_data[:split_idx]
    test_data = all_arch_data[split_idx:]

    train_loader = DataLoader(DentalArchDataset(train_data, is_train=True), batch_size=16, shuffle=True)
    test_loader = DataLoader(DentalArchDataset(test_data, is_train=False), batch_size=16, shuffle=False)

    # ==========================================
    # 2. 初始化模型与动态学习率
    # ==========================================
    model = MaskedArchRegressor().to(device)

    # 初始学习率从 1e-3 下调到 3e-4，防止大模型震荡
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # 引入调度器：如果 Test Loss 连续 10 次不下降，学习率自动减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion = nn.MSELoss(reduction='none')

    epochs = 1200  # 可以跑 1200 轮，有了 scheduler 不怕跑飞
    best_test_loss = float('inf')
    save_path = "best_curve_model.pth"

    # ==========================================
    # 3. 训练循环
    # ==========================================
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            gt_centers = batch["gt_centers"].to(device)
            dropped_mask = batch["dropped_mask"].to(device)

            optimizer.zero_grad()
            preds = model(features)

            loss_matrix = criterion(preds, gt_centers)
            loss_per_tooth = loss_matrix.mean(dim=-1)

            valid_loss = loss_per_tooth[dropped_mask]

            if len(valid_loss) > 0:
                loss = valid_loss.mean()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

        # ----------------------------------------
        # 验证循环
        # ----------------------------------------
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_loss = 0.0
            test_batches = 0

            with torch.no_grad():
                for batch in test_loader:
                    test_centers = batch["gt_centers"].clone()
                    test_valid_mask = batch["valid_mask"]

                    # 强行掩码测试集中几个随机的牙齿进行评估
                    test_dropped_mask = torch.zeros_like(test_valid_mask)
                    for i in range(test_valid_mask.shape[0]):
                        valid_idx = torch.where(test_valid_mask[i])[0]
                        if len(valid_idx) > 2:
                            drop_idx = random.sample(valid_idx.tolist(), 2)
                            test_dropped_mask[i, drop_idx] = True

                    test_centers[test_dropped_mask] = 0.0
                    is_missing = (~test_valid_mask | test_dropped_mask).float()

                    # 【修复点】：从原始 batch 中提取出 jaw_type (它在最后一列，索引为 4)
                    jaw_type = batch["features"][:, :, 4:5]

                    # 【修复点】：将 jaw_type 拼接到最后，凑齐 5 维 (X, Y, Z, missing, jaw)
                    test_features = torch.cat([test_centers, is_missing.unsqueeze(-1), jaw_type], dim=-1).to(device)

                    preds = model(test_features)
                    t_loss_matrix = criterion(preds, batch["gt_centers"].to(device)).mean(dim=-1)
                    t_valid_loss = t_loss_matrix[test_dropped_mask.to(device)]

                    if len(t_valid_loss) > 0:
                        test_loss += t_valid_loss.mean().item()
                        test_batches += 1



            avg_test_loss = test_loss / test_batches if test_batches > 0 else 0.0

            # 让调度器根据测试集 Loss 决定是否踩刹车
            scheduler.step(avg_test_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # 保存最佳模型
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), save_path)
                status = "<-- New Best! Saved."
            else:
                status = ""

            # 打印时顺便把当前的学习率打出来，你能看到它在后期自动变小
            print(
                f"Epoch {epoch + 1:03d} | LR: {current_lr:.1e} | Train Loss: {total_train_loss / len(train_loader):.4f} | Test Loss: {avg_test_loss:.4f} {status}")

            # 保存最佳模型
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), save_path)
                status = "<-- New Best! Saved."
            else:
                status = ""

            print(
                f"Epoch {epoch + 1:03d} | Train Loss: {total_train_loss / len(train_loader):.4f} | Test Loss: {avg_test_loss:.4f} {status}")


if __name__ == "__main__":
    main()