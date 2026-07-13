import os
import json


def analyze_landmark_completeness(root_path):
    # 定义项目所必须的 4 个核心面 (映射到 JSON 中的 class 名称)
    REQUIRED_LABELS = {"Mesial", "Distal", "InnerPoint", "OuterPoint"}

    satisfied_count = 0
    unsatisfied_count = 0

    if not os.path.exists(root_path):
        print(f"❌ 错误：找不到路径 {root_path}")
        return

    # 遍历所有的 ID_single 文件夹
    for item in os.listdir(root_path):
        folder_path = os.path.join(root_path, item)

        if os.path.isdir(folder_path) and item.endswith("_single"):
            # 找到该文件夹下的所有 json 文件
            json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]

            for json_file in json_files:
                json_path = os.path.join(folder_path, json_file)

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 提取这颗牙齿拥有的所有标签类别
                    # 注意：之前我们存入的格式是 [{"class": "Mesial", ...}, ...]
                    current_labels = set([pt['class'] for pt in data])

                    # 判断 REQUIRED_LABELS 是否为 current_labels 的子集
                    if REQUIRED_LABELS.issubset(current_labels):
                        satisfied_count += 1
                    else:
                        unsatisfied_count += 1

                except Exception as e:
                    print(f"读取文件 {json_file} 失败: {e}")

    # 打印最终统计报告
    print("=" * 50)
    print(f"🎯 单齿面标注完整度统计报告")
    print("-" * 50)
    print(f"✅ 满足条件 (含 Mesial, Distal, Inner, Outer): {satisfied_count} 颗")
    print(f"⚠️ 不满足条件 (缺失上述任意一个或多个面):    {unsatisfied_count} 颗")
    print(f"总计检查牙齿数: {satisfied_count + unsatisfied_count} 颗")
    print("=" * 50)


if __name__ == "__main__":
    DATA_PATH = r"F:\NDCS_3DS_data\single_teeth_dataset_train"
    analyze_landmark_completeness(DATA_PATH)