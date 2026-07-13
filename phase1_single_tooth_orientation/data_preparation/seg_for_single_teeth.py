import os
import json
import numpy as np
import trimesh
from scipy.spatial import KDTree

# 定义 Mentor 数据集中的 5 个标准方向面
STANDARD_CLASSES = ["Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint"]


def compute_distance_maps(mesh, landmarks, centroid):
    """
    计算五维距离图特征 (5, num_faces)
    分别计算面片中心到 Mesial, Distal, Inner, Outer, Facial 五个点的欧式距离。
    这是 PTv3 模型训练真正的 Ground Truth。
    """
    # 计算每个面片的几何中心 (num_faces, 3)
    face_centers = mesh.triangles.mean(axis=1)

    num_faces = len(mesh.faces)
    dist_maps = np.zeros((5, num_faces), dtype=np.float64)

    # 建立类别到坐标的映射
    lm_dict = {lm['class']: np.array(lm['coord']) for lm in landmarks}

    # 按顺序计算 5 维距离
    for i, cls_name in enumerate(STANDARD_CLASSES):
        if cls_name in lm_dict:
            # 计算面片中心到该地标的欧式距离
            target_coord = lm_dict[cls_name]
            distances = np.linalg.norm(face_centers - target_coord, axis=1)
            dist_maps[i, :] = distances
        else:
            # 如果这颗牙确实缺失了某个方向的标注，填入一个较大的默认距离(如 20.0mm)
            # 这样在算 Loss 时可以根据阈值屏蔽掉这部分，或者起到惩罚作用
            dist_maps[i, :] = 20.0

    return dist_maps


def process_single_arch(obj_path, label_path, kpt_path, output_folder, patient_id, jaw):
    """处理单个牙弓，生成若干单颗牙齿的 OFF, JSON, NPY"""
    try:
        # 1. 加载数据
        mesh = trimesh.load(obj_path)
        with open(label_path, 'r') as f:
            labels = np.array(json.load(f)['labels'])
        with open(kpt_path, 'r') as f:
            landmarks = json.load(f).get('objects', [])

        if len(labels) != len(mesh.vertices):
            print(f"  [警告] {patient_id}_{jaw}: 顶点数({len(mesh.vertices)})与标签数({len(labels)})不匹配，跳过！")
            return

        # 2. KDTree 绑定地标到牙位
        kdtree = KDTree(mesh.vertices)
        landmark_mapping = []
        for lm in landmarks:
            coord = np.array(lm['coord'])
            _, idx = kdtree.query(coord)
            tooth_id = labels[idx]
            if tooth_id != 0:  # 排除掉落在牙龈(0)上的异常点
                landmark_mapping.append({'tooth_id': tooth_id, 'lm_data': lm})

        # 3. 提取每颗单牙
        unique_teeth = np.unique(labels)
        unique_teeth = unique_teeth[unique_teeth != 0]

        for tooth_id in unique_teeth:
            # --- A. 提取单齿网格 ---
            vertex_mask = (labels == tooth_id)
            face_mask = np.all(vertex_mask[mesh.faces], axis=1)
            tooth_mesh = mesh.submesh([face_mask], append=True)

            if len(tooth_mesh.vertices) < 50:
                continue  # 过滤掉分割产生的极小碎片

            # --- B. 归一化 (平移至原点) ---
            centroid = tooth_mesh.vertices.mean(axis=0)
            tooth_mesh.vertices -= centroid

            # --- C. 提取并归一化地标 ---
            tooth_lms = []
            for item in landmark_mapping:
                if item['tooth_id'] == tooth_id:
                    lm = item['lm_data']
                    norm_coord = np.array(lm['coord']) - centroid
                    tooth_lms.append({
                        "key": lm['key'],
                        "class": lm['class'],
                        "coord": norm_coord.tolist()
                    })

            if not tooth_lms:
                continue  # 过滤掉完全没有匹配到地标的牙齿

            # --- D. 计算 5 维距离图 ---
            features = compute_distance_maps(tooth_mesh, tooth_lms, centroid)

            # --- E. 导出三个文件 ---
            prefix = f"{patient_id}_{jaw}_tooth_{tooth_id}"

            # 保存 OFF
            tooth_mesh.export(os.path.join(output_folder, f"{prefix}.off"))
            # 保存 JSON
            with open(os.path.join(output_folder, f"{prefix}.json"), 'w') as f:
                json.dump(tooth_lms, f, indent=4)
            # 保存 NPY
            np.save(os.path.join(output_folder, f"{prefix}_f.npy"), features)

    except Exception as e:
        print(f"  [错误] 处理 {patient_id}_{jaw} 时发生异常: {e}")


def batch_process_dataset(seg_root_dir, kpt_root_dir, output_root):
    """
    批量遍历，分别从分割目录和地标目录抓取对应文件
    """
    jaws = ['upper', 'lower']
    total_processed = 0

    os.makedirs(output_root, exist_ok=True)

    for jaw in jaws:
        jaw_seg_dir = os.path.join(seg_root_dir, jaw)
        jaw_kpt_dir = os.path.join(kpt_root_dir, jaw)

        if not os.path.exists(jaw_seg_dir):
            print(f"找不到分割数据文件夹: {jaw_seg_dir}")
            continue

        print(f"\n🚀 开始处理 {jaw} 牙弓...")

        # 遍历 jaw_seg_dir 下的每一个病人文件夹 (例如: 01A6GW4A)
        for patient_id in os.listdir(jaw_seg_dir):
            patient_seg_dir = os.path.join(jaw_seg_dir, patient_id)

            if not os.path.isdir(patient_seg_dir):
                continue

            # 1. 拼接【分割数据】的绝对路径
            obj_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw}.obj")
            label_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw}.json")

            # 2. 拼接【地标数据】的绝对路径
            patient_kpt_dir = os.path.join(jaw_kpt_dir, patient_id)
            kpt_path = os.path.join(patient_kpt_dir, f"{patient_id}_{jaw}__kpt.json")

            # 3. 严格检查三个文件是否全都存在
            if not os.path.exists(obj_path):
                print(f"[跳过] {patient_id}: 找不到 OBJ 文件")
                continue
            if not os.path.exists(label_path):
                print(f"[跳过] {patient_id}: 找不到 标签 文件")
                continue
            if not os.path.exists(kpt_path):
                print(f"[跳过] {patient_id}: 找不到 地标 文件")
                continue

            # 4. 创建专门的输出文件夹: 01A6GW4A_single
            output_folder = os.path.join(output_root, f"{patient_id}_single")
            os.makedirs(output_folder, exist_ok=True)

            # 5. 执行切分
            process_single_arch(obj_path, label_path, kpt_path, output_folder, patient_id, jaw)
            total_processed += 1

            if total_processed % 10 == 0:
                print(f"已处理 {total_processed} 个牙弓样本...")

    print(f"\n✅ 批量分割完成！")
    print(f"共成功读取了 {total_processed} 个全牙弓扫描。")
    print(f"所有单颗牙齿的独立数据已保存在: {output_root}")


if __name__ == "__main__":
    # ================= 配置路径 =================
    # 1. 分割数据根目录 (包含 .obj 和 带 labels 数组的 .json)
    SEG_ROOT = r"F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train"

    # 2. 地标数据根目录 (包含 __kpt.json)
    KPT_ROOT = r"F:\NDCS_3DS_data\3DTeethLand_landmarks_train"

    # 3. 最终单颗牙齿输出的保存根目录
    OUTPUT_ROOT = r"F:\NDCS_3DS_data\single_teeth_dataset_train"
    # ============================================

    batch_process_dataset(SEG_ROOT, KPT_ROOT, OUTPUT_ROOT)