import os
import shutil
import random
import json


def split_dataset_by_patient(src_dir, dest_dir, train_ratio=0.85):
    """
    按病人/牙弓级别进行 Train/Test 划分，并统计各自的完整度
    """
    # 固定随机种子，保证每次运行划分结果一致
    random.seed(42)

    # 1. 获取所有病人文件夹
    patient_folders = [f for f in os.listdir(src_dir)
                       if os.path.isdir(os.path.join(src_dir, f)) and f.endswith("_single")]

    # 2. 随机打乱并划分
    random.shuffle(patient_folders)
    split_idx = int(len(patient_folders) * train_ratio)

    train_folders = patient_folders[:split_idx]
    test_folders = patient_folders[split_idx:]

    # 创建目标目录
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def copy_and_count(folders, target_dir, split_name):
        print(f"正在复制并统计 {split_name} 集数据...")
        total_teeth = 0
        complete_teeth = 0
        REQUIRED_LABELS = {"Mesial", "Distal", "InnerPoint", "OuterPoint"}

        for folder in folders:
            src_folder_path = os.path.join(src_dir, folder)
            dest_folder_path = os.path.join(target_dir, folder)
            os.makedirs(dest_folder_path, exist_ok=True)

            files = os.listdir(src_folder_path)
            for file in files:
                # 复制文件到新目录
                shutil.copy2(os.path.join(src_folder_path, file),
                             os.path.join(dest_folder_path, file))

            # 统计完整度
            json_files = [f for f in files if f.endswith('.json')]
            total_teeth += len(json_files)
            for jf in json_files:
                with open(os.path.join(src_folder_path, jf), 'r') as f:
                    data = json.load(f)
                    labels = set([pt['class'] for pt in data])
                    if REQUIRED_LABELS.issubset(labels):
                        complete_teeth += 1

        return len(folders), total_teeth, complete_teeth

    # 3. 执行复制与统计
    print("🚀 开始按牙弓级别严格隔离划分数据...\n")
    tr_folds, tr_tot, tr_comp = copy_and_count(train_folders, train_dir, "Train")
    te_folds, te_tot, te_comp = copy_and_count(test_folders, test_dir, "Test")

    print("\n" + "=" * 50)
    print("🎯 数据集划分完成报告")
    print("=" * 50)
    print(f"📂 训练集 (Train): {tr_folds} 个牙弓, 共 {tr_tot} 颗单牙")
    print(f"   -> 完整标注: {tr_comp} 颗 | 缺失标注: {tr_tot - tr_comp} 颗")
    print("-" * 50)
    print(f"📂 测试集 (Test):  {te_folds} 个牙弓, 共 {te_tot} 颗单牙")
    print(f"   -> 完整标注: {te_comp} 颗 | 缺失标注: {te_tot - te_comp} 颗")
    print("=" * 50)
    print(f"数据已成功保存至: {dest_dir}")


if __name__ == "__main__":
    SRC_PATH = r"F:\NDCS_3DS_data\single_teeth_dataset_train"
    DEST_PATH = r"F:\NDCS_3DS_data\Single_Teeth_Y"

    split_dataset_by_patient(SRC_PATH, DEST_PATH)