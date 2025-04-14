import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import os

# ------------------- 设置页面标题和布局 -------------------
st.set_page_config(page_title="🧠 ML + EDA Demo", layout="centered")
st.title("🧪 Streamlit 数据科学全流程 Demo")

# ------------------- Session 状态初始化 -------------------
# 初始化 session 状态，记录模型是否训练过
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

st.subheader("📁 数据加载")
# 上传 CSV 文件，如果没有上传则使用内置 iris 数据集
uploaded_file = st.file_uploader("上传你的 CSV 文件，或者使用内置 Iris 数据集：", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    df['target'] = df['target'].astype(int)

# 展示数据前几行
st.dataframe(df.head())

# ------------------- 2. EDA 图表展示 -------------------
st.subheader("📊 数据可视化")
# 选择数值列用于绘图
selected_col = st.selectbox(
    "选择一个数值列绘图：", df.select_dtypes(include=np.number).columns)

# 选择图表类型
chart_type = st.radio("选择图表类型", ["直方图", "折线图"])
fig, ax = plt.subplots()
if chart_type == "直方图":
    ax.hist(df[selected_col], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f"{selected_col} - 直方图")
else:
    ax.plot(df[selected_col], color='green')
    ax.set_title(f"{selected_col} - 折线图")
st.pyplot(fig)

# ------------------- 3. 模型训练或加载 -------------------
st.subheader("🤖 模型训练 / 加载")

# 模型训练设置面板
with st.expander("模型训练设置"):
    test_size = st.slider("测试集占比", 0.1, 0.5, 0.2, step=0.05)
    n_estimators = st.slider("随机森林树数", 10, 200, 100, step=10)
    train_button = st.button("点击训练模型")  # 点击训练按钮

# 如果用户点击了训练按钮，则进行训练
if train_button:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 保存模型
    joblib.dump(model, "model.pkl")
    st.session_state.model_trained = True  # 更新训练状态
    st.success(f"✅ 模型训练完成，测试准确率：{acc:.2f}")

# ------------------- 4. 模型预测 -------------------
st.subheader("📍 在线模型预测")
# 如果模型训练过或已存在模型文件，则允许预测
if st.session_state.model_trained or os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
    st.write("👉 输入特征进行预测：")
    input_values = []
    # 构造输入表单，逐列输入数值
    for col in df.columns[:-1]:
        val = st.number_input(f"{col}", float(df[col].min()), float(
            df[col].max()), float(df[col].mean()))
        input_values.append(val)
    # 点击预测按钮触发预测
    if st.button("预测！"):
        pred = model.predict([input_values])[0]
        st.success(f"🎉 预测结果: 类别 {pred}")
else:
    st.info("请先训练模型或上传模型文件")


# ------------------- 5. 部署说明 -------------------
st.markdown("---")
st.caption("📤 部署建议：使用 `streamlit share` 或上传到 HuggingFace Spaces / Render")
