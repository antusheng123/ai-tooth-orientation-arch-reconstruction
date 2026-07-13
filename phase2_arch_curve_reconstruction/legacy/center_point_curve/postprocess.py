import numpy as np


def generate_smooth_curve(points, num_eval_points=100, k=None):
    """
    使用二次多项式(抛物线)逼近 3D 关键点，生成完美的 U 形牙弓曲线，彻底消除外弧波动。
    """
    valid_mask = np.linalg.norm(points, axis=1) > 1e-5
    valid_points = points[valid_mask]

    if len(valid_points) < 4:
        return valid_points

    x = valid_points[:, 0]
    y = valid_points[:, 1]
    z = valid_points[:, 2]

    # 1. 在 X-Y 平面进行二次抛物线拟合 (保证 U 形基底)
    # np.polyfit 返回多项式系数 [a, b, c]，对应 y = a*x^2 + b*x + c
    poly_xy = np.polyfit(x, y, deg=2)

    # 2. 在 X-Z 平面进行二次拟合 (保留合曲线的 Curve of Spee 高低起伏)
    poly_xz = np.polyfit(x, z, deg=2)

    # 3. 生成平滑的新点
    # 按照 X 轴的最小值和最大值均匀采样
    x_fine = np.linspace(x.min() - 1.0, x.max() + 1.0, num_eval_points)

    # 根据拟合出来的抛物线方程计算对应的 Y 和 Z
    y_fine = np.polyval(poly_xy, x_fine)
    z_fine = np.polyval(poly_xz, x_fine)

    # 组合输出
    curve_points = np.vstack((x_fine, y_fine, z_fine)).T

    return curve_points