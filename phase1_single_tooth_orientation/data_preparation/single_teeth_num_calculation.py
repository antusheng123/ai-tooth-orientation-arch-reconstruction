import os


def count_total_extracted_teeth(root_path):
    """
    遍历目标路径，统计所有子文件夹中 .off 文件的总数
    """
    total_teeth = 0
    folder_count = 0

    if not os.path.exists(root_path):
        print(f"❌ 错误：找不到路径 {root_path}")
        return

    # 遍历根目录下的所有子文件夹
    for item in os.listdir(root_path):
        folder_path = os.path.join(root_path, item)

        # 只处理以 _single 结尾的文件夹
        if os.path.isdir(folder_path) and item.endswith("_single"):
            folder_count += 1
            # 统计该文件夹下后缀为 .off 的文件数量
            off_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.off')]
            total_teeth += len(off_files)

    print("=" * 40)
    print(f"📊 数据统计报告")
    print("-" * 40)
    print(f"📂 扫描到的病人(牙弓)文件夹总数: {folder_count}")
    print(f"🦷 成功分割出的独立牙齿总数:   {total_teeth}")
    print("=" * 40)

    return total_teeth


if __name__ == "__main__":
    # 你的数据存储路径
    DATA_PATH = r"F:\NDCS_3DS_data\single_teeth_dataset_train"

    count_total_extracted_teeth(DATA_PATH)