import os
import glob
import numpy as np
import trimesh
import shutil
from tqdm import tqdm

'''
def pre_sample_to_new_root(src_root, dst_root, num_points=4096):

    # 1. 递归寻找所有单齿 off 文件
    off_files = glob.glob(os.path.join(src_root, "**", "*.off"), recursive=True)
    print(f"🚀 找到 {len(off_files)} 个模型，正在构建点云数据集到: {dst_root}")

    for off_path in tqdm(off_files):
        # 2. 计算相对路径，以便在目标目录维持相同的 train/test/ID_single 结构
        rel_path = os.path.relpath(off_path, src_root)

        # 3. 确定目标文件路径
        # 点云路径: .../Single_Teeth_PC/train/ID_single/xxx_pc2048.npy
        dst_npy_path = os.path.join(dst_root, rel_path).replace(".off", "_pc2048.npy")
        # 标签路径: .../Single_Teeth_PC/train/ID_single/xxx.json (直接拷贝过去)
        src_json_path = off_path.replace(".off", ".json")
        dst_json_path = os.path.join(dst_root, rel_path).replace(".off", ".json")

        # 创建目标文件夹
        os.makedirs(os.path.dirname(dst_npy_path), exist_ok=True)

        # 4. 执行采样并保存
        if not os.path.exists(dst_npy_path):
            try:
                mesh = trimesh.load(off_path, process=False)
                points, _ = trimesh.sample.sample_surface(mesh, num_points)
                np.save(dst_npy_path, points.astype(np.float32))
            except Exception as e:
                print(f"❌ 采样失败 {off_path}: {e}")
                continue

        # 5. 顺便把对应的 JSON 标签文件拷贝过去，这样新目录就是自包含的训练集
        if os.path.exists(src_json_path) and not os.path.exists(dst_json_path):
            shutil.copy2(src_json_path, dst_json_path)

    print(f"\n✅ 转换完成！点云训练集位于: {dst_root}")


if __name__ == "__main__":
    SRC = r"PATH"
    DST = r"PATH"
    pre_sample_to_new_root(SRC, DST)
'''
import os
import glob
import numpy as np
import trimesh
import shutil
from tqdm import tqdm


def pre_sample_6d(src_root, dst_root, num_points=4096):
    off_files = glob.glob(os.path.join(src_root, "**", "*.off"), recursive=True)
    print(f"🚀 准备生成带有【法向量】的 6D 点云，共 {len(off_files)} 个模型...")

    for off_path in tqdm(off_files):
        rel_path = os.path.relpath(off_path, src_root)
        dst_npy_path = os.path.join(dst_root, rel_path).replace(".off", "_pc4096_normals.npy")
        src_json_path = off_path.replace(".off", ".json")
        dst_json_path = os.path.join(dst_root, rel_path).replace(".off", ".json")

        os.makedirs(os.path.dirname(dst_npy_path), exist_ok=True)

        if not os.path.exists(dst_npy_path):
            try:
                mesh = trimesh.load(off_path, process=False)
                # 1. 采样表面点，同时返回被采样的面片索引 (face_indices)
                points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

                # 2. 根据面片索引，提取对应的面法向量
                normals = mesh.face_normals[face_indices]

                # 3. 拼接成 (N, 6) 的数组: [x, y, z, nx, ny, nz]
                points_with_normals = np.concatenate([points, normals], axis=1)

                np.save(dst_npy_path, points_with_normals.astype(np.float32))
            except Exception as e:
                print(f"❌ 采样失败 {off_path}: {e}")
                continue

        if os.path.exists(src_json_path) and not os.path.exists(dst_json_path):
            shutil.copy2(src_json_path, dst_json_path)

    print(f"\n✅ 6D 点云生成完毕！保存在: {dst_root}")


if __name__ == "__main__":
    # 建议将 dst 换个新名字，比如 Single_Teeth_PC_6D
    SRC = r"F:\NDCS_3DS_data\Single_Teeth_Y"
    DST = r"F:\NDCS_3DS_data\Single_Teeth_PC_6D"
    pre_sample_6d(SRC, DST)