import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import os

# 导入已经写好的模块特征与全局配置
from dataset import extract_arch_centers, UPPER_FDI, LOWER_FDI, SEQ_LEN
from model import MaskedArchRegressor
from postprocess import generate_smooth_curve

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="Dental Arch AI Curve", layout="wide")
st.title("🦷 Dental Arch AI Completion & Curve Generation System (Upper & Lower)")
st.markdown("Based on Upgraded Masked Transformer & Parabolic Approximation Fitting")

# 真实根目录路径 (更新为你的统一父级路径)
SEG_ROOT = r"F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train"
KPT_ROOT = r"F:\NDCS_3DS_data\3DTeethLand_landmarks_train"


# ==========================================
# 1. 缓存模型加载（秒开，只加载一次）
# ==========================================
@st.cache_resource(show_spinner="Initializing AI Model...")
def load_model_weights():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedArchRegressor().to(device)
    model_path = "best_curve_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        st.error(f"❌ Cannot find model weights {model_path}, please run train.py first!")
        return None, device


model, device = load_model_weights()


# ==========================================
# 2. 缓存单人数据的提取（支持跨子文件夹动态定位）
# ==========================================
@st.cache_data(show_spinner=None)
def get_single_patient_data(option_string):
    # 从选项字符串中拆分出病人ID和牙颌类别 (例如从 '01A6GW4A_upper' 拆分)
    patient_id, jaw_name = option_string.rsplit('_', 1)
    jaw_val = 0 if jaw_name == 'upper' else 1

    patient_seg_dir = os.path.join(SEG_ROOT, jaw_name, patient_id)
    patient_kpt_dir = os.path.join(KPT_ROOT, jaw_name, patient_id)

    obj_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.obj")
    label_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.json")
    kpt_path = os.path.join(patient_kpt_dir, f"{patient_id}_{jaw_name}__kpt.json")

    if os.path.exists(obj_path) and os.path.exists(label_path) and os.path.exists(kpt_path):
        try:
            # 引入了我们更新过带有 4 个参数的中心点提取函数
            centers, valid_mask = extract_arch_centers(obj_path, label_path, kpt_path, jaw_val)
            return centers, valid_mask
        except Exception as e:
            return None, str(e)
    return None, f"Cannot find the corresponding file triad"


# ==========================================
# 3. 动态且安全地检索所有上下颌病人样本
# ==========================================
all_patient_options = []
jaw_types = ['upper', 'lower']

if os.path.exists(SEG_ROOT) and os.path.exists(KPT_ROOT):
    for jaw_name in jaw_types:
        seg_jaw_dir = os.path.join(SEG_ROOT, jaw_name)
        kpt_jaw_dir = os.path.join(KPT_ROOT, jaw_name)

        if os.path.exists(seg_jaw_dir) and os.path.exists(kpt_jaw_dir):
            for p in os.listdir(seg_jaw_dir):
                p_seg_path = os.path.join(seg_jaw_dir, p)
                p_kpt_path = os.path.join(kpt_jaw_dir, p)

                if os.path.isdir(p_seg_path) and os.path.exists(p_kpt_path):
                    # 提前做一次文件存在验证，确保加进菜单的都是完好的样本
                    obj_path = os.path.join(p_seg_path, f"{p}_{jaw_name}.obj")
                    if os.path.exists(obj_path):
                        all_patient_options.append(f"{p}_{jaw_name}")

    all_patient_options = sorted(all_patient_options)
else:
    all_patient_options = []

if not model or not all_patient_options:
    st.error("🚨 Path configuration error or no valid data detected. Please check local F drive paths.")
    st.stop()

# ==========================================
# 侧边栏：交互式控制
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Panel")
    selected_patient = st.selectbox("📂 Select Patient Arch Sample", all_patient_options)

with st.spinner(f"Parsing 3D mesh features for patient {selected_patient} in real-time..."):
    raw_centers, valid_mask_or_err = get_single_patient_data(selected_patient)

if raw_centers is None:
    st.error(f"❌ Failed to load patient data: {valid_mask_or_err}")
    st.stop()

valid_mask = valid_mask_or_err

# 解析当前所选样本的牙位基准 (根据是 upper 还是 lower 动态切换 FDI 系统)
_, selected_jaw_name = selected_patient.rsplit('_', 1)
CURRENT_FDI = UPPER_FDI if selected_jaw_name == 'upper' else LOWER_FDI

# ==========================================
# 侧边栏补充：动态生成缺牙选项
# ==========================================
with st.sidebar:
    st.divider()
    st.subheader("🛠️ Simulate Missing Teeth (Masking)")
    st.write("Please select the teeth to manually 'extract'. AI will predict their positions based on the remaining ones:")

    existing_fdi = [CURRENT_FDI[i] for i in range(SEQ_LEN) if valid_mask[i]]
    teeth_to_drop = st.multiselect("Select teeth to mask (FDI):", existing_fdi)

# ==========================================
# 核心推理逻辑
# ==========================================
input_centers = raw_centers.copy()
dropped_mask = np.zeros(SEQ_LEN, dtype=bool)

for fdi in teeth_to_drop:
    idx = CURRENT_FDI.index(fdi)
    dropped_mask[idx] = True
    input_centers[idx] = [0.0, 0.0, 0.0]

# 组装 5 维特征所需的数据
jaw_val = 0 if selected_jaw_name == 'upper' else 1
jaw_feature = np.full((SEQ_LEN, 1), jaw_val, dtype=np.float32)
is_missing = (~valid_mask | dropped_mask).astype(np.float32)

# 完美契合训练模型的 5 维特征拼接
features = np.concatenate([input_centers, is_missing[:, None], jaw_feature], axis=-1)
feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(feat_tensor)
    pred_centers = preds[0].cpu().numpy()

# 筛选并隔离天然缺失的牙齿
final_centers_for_curve = []
exist_x, exist_y, exist_z, exist_text = [], [], [], []
pred_x, pred_y, pred_z, pred_text = [], [], [], []

for i in range(SEQ_LEN):
    if valid_mask[i]:  # 只针对原本就存在的牙齿进行连线和统计
        if not dropped_mask[i]:
            coord = raw_centers[i]
            final_centers_for_curve.append(coord)
            exist_x.append(coord[0])
            exist_y.append(coord[1])
            exist_z.append(coord[2])
            exist_text.append(str(CURRENT_FDI[i]))
        else:
            coord = pred_centers[i]
            final_centers_for_curve.append(coord)
            pred_x.append(coord[0])
            pred_y.append(coord[1])
            pred_z.append(coord[2])
            pred_text.append(str(CURRENT_FDI[i]) + " (AI Pred)")

final_centers_for_curve = np.array(final_centers_for_curve)

# 使用抛物线拟合后处理生成完美的 U 形曲线
curve_points = generate_smooth_curve(final_centers_for_curve, num_eval_points=200)

# ==========================================
# Plotly 3D 渲染展示
# ==========================================
col1, col2 = st.columns([1, 3])

with col1:
    st.info(f"**Current Sample:** {selected_patient}")
    st.metric("Retained Real Teeth Count", len(exist_x))
    st.metric("Simulated Missing Teeth Count (AI Pred)", len(pred_x))

    st.write("💡 **Legend Description**")
    st.markdown("<span style='color: rgb(54, 162, 235); font-size: 20px;'>●</span> Intact Landmarks",
                unsafe_allow_html=True)
    st.markdown("<span style='color: rgb(255, 99, 132); font-size: 20px;'>♦</span> AI Imputed Landmarks",
                unsafe_allow_html=True)
    st.markdown("<span style='color: rgb(75, 192, 192); font-size: 20px;'>▬</span> Fitted Arch Curve",
                unsafe_allow_html=True)

with col2:
    fig = go.Figure()

    # 1. 真实点
    if exist_x:
        fig.add_trace(go.Scatter3d(
            x=exist_x, y=exist_y, z=exist_z,
            mode='markers+text',
            marker=dict(size=8, color='rgb(54, 162, 235)', line=dict(width=2, color='white')),
            text=exist_text, textposition="top center", name='Intact'
        ))

    # 2. 预测点
    if pred_x:
        fig.add_trace(go.Scatter3d(
            x=pred_x, y=pred_y, z=pred_z,
            mode='markers+text',
            marker=dict(size=10, symbol='diamond', color='rgb(255, 99, 132)', line=dict(width=2, color='white')),
            text=pred_text, textposition="top center", name='AI Pred'
        ))

    # 3. 抛物线拟合曲线
    if len(final_centers_for_curve) > 0:
        fig.add_trace(go.Scatter3d(
            x=curve_points[:, 0], y=curve_points[:, 1], z=curve_points[:, 2],
            mode='lines',
            line=dict(color='rgb(75, 192, 192)', width=6),
            hoverinfo='skip', name='Fitted Curve'
        ))

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0), height=700, showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)