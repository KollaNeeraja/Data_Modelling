# ------------------- Imports ---------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.stats as stats

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import estimator_html_repr
import streamlit.components.v1 as components
from graphviz import Digraph
from sklearn.base import BaseEstimator

shap.initjs()

# ------------------- Streamlit Config ---------------------
st.set_page_config(layout="wide", page_title="📊 Stacking Regressor App", initial_sidebar_state="expanded")
st.image("https://upload.wikimedia.org/wikipedia/commons/6/6a/Stacking_icon.png", width=100)
st.title("🔍 Predictive Modeling with Stacking Regressor")

# ------------------- Sidebar ---------------------
st.sidebar.header("📄 Upload or Use Default Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully.")
else:
    df = pd.read_csv("C:/Users/Neeraja/Downloads/Data Scientist_Model Test Data.csv")
    st.sidebar.info("ℹ️ Using default dataset")

# ------------------- Preprocess ---------------------
X = df.drop(columns='y')
y = df['y']
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_full, y_train_full)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_features = feature_importance['Feature'].values[:3]
X_top = df[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

# ------------------- Pipeline ---------------------
base_models = [
    ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)),
    ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
]
meta_model = LassoCV(alphas=[0.01, 0.1, 1.0], cv=5, max_iter=10000, n_jobs=-1)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        passthrough=True
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_score = cross_val_score(pipeline, X_top, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1).mean()
residuals = y_test - y_pred

# ------------------- Tabs ---------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset", 
    "📈 Modeling", 
    "🧠 Explainability", 
    "📅 Download", 
    "📘 Pipeline HTML"
])

# ------------------- Tab 1: Dataset ---------------------
with tab1:
    st.subheader("📊 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.subheader("🔗 Correlation Matrix")
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

# ------------------- Tab 2: Modeling ---------------------
def build_pipeline_graph(pipeline, title="Model Pipeline Overview"):
    dot = Digraph(format='png', graph_attr={'rankdir': 'TB', 'splines': 'spline'})
    dot.attr(label=title, labelloc="top", fontsize='20', fontname='Helvetica')

    styles = {
        "scaler": "#a3d2ca",
        "base": "#ffcda3",
        "meta": "#f6dfeb",
        "stacking": "#d3e4cd"
    }

    dot.node("Scaler", "🔹 StandardScaler", style="filled", fillcolor=styles["scaler"], shape="box")
    dot.node("Ridge", "🔸 RidgeCV", style="filled", fillcolor=styles["base"], shape="box")
    dot.node("GB", "🔸 GradientBoosting", style="filled", fillcolor=styles["base"], shape="box")
    dot.node("Lasso", "🔶 LassoCV (Meta)", style="filled", fillcolor=styles["meta"], shape="box")
    dot.node("Stacking", "🧱 StackingRegressor", style="filled", fillcolor=styles["stacking"], shape="box")

    dot.edge("Scaler", "Ridge", label="→")
    dot.edge("Scaler", "GB", label="→")
    dot.edge("Ridge", "Stacking")
    dot.edge("GB", "Stacking")
    dot.edge("Lasso", "Stacking", style="dashed", label="Meta Learner")

    return dot

with tab2:
    st.subheader("🧠 Visual Model Architecture")

    dot = Digraph(format='png', graph_attr={'rankdir': 'TB', 'splines': 'true'})
    dot.attr('node', shape='box', style='filled', fontname='Helvetica')

    dot.node("Scaler", "🔹 StandardScaler", fillcolor="#cce5ff")
    dot.node("Ridge", "🔸 RidgeCV\n(alphas=[0.1, 1.0, 10.0], cv=5)", fillcolor="#ffe4b3")
    dot.node("GB", "🔸 GradientBoosting\n(n_estimators=100,\nmax_depth=3)", fillcolor="#ffe4b3")
    dot.node("Lasso", "🔶 LassoCV\n(alphas=[0.01, 0.1, 1.0], cv=5)", fillcolor="#d4edda")
    dot.node("Stacking", "🧱 StackingRegressor", fillcolor="#e2e3e5")

    dot.edge("Scaler", "Stacking", label="fit/predict", fontsize="10")
    dot.edge("Ridge", "Stacking")
    dot.edge("GB", "Stacking")
    dot.edge("Stacking", "Lasso", label="meta-learning", style="dashed", fontsize="10")

    st.graphviz_chart(dot)

    st.subheader("🌲 Feature Importances")
    fig_feat, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    st.pyplot(fig_feat)
    st.success(f"Top 3 Features Used: {', '.join(top_features)}")

    st.subheader("📊 Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("Test R²", f"{r2:.4f}")
    col3.metric("CV R²", f"{cv_score:.4f}")

    

    st.subheader("🔍 Prediction vs Actual")
    fig_pred, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig_pred)

    st.subheader("🔍 Residuals")
    fig_resid, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    st.pyplot(fig_resid)

    fig_hist, ax = plt.subplots()
    sns.histplot(residuals, bins=20, kde=True, ax=ax)
    st.pyplot(fig_hist)

# ------------------- Tab 3: Explainability ---------------------
with tab3:
    st.subheader("🔍 SHAP Interpretability")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    explainer = shap.Explainer(gb_model, X_train)
    shap_values = explainer(X_test)

    st.markdown("**Global Feature Importance (SHAP Bar Plot)**")
    shap.plots.bar(shap_values, show=False)
    st.pyplot(plt.gcf())

    st.markdown("**SHAP Summary (Beeswarm)**")
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(plt.gcf())

    st.markdown("**Waterfall Plot (Random Sample)**")
    idx = np.random.randint(0, X_test.shape[0])
    shap.plots.waterfall(shap_values[idx], show=False)
    st.pyplot(plt.gcf())

    st.subheader("📊 QQ Plot of Residuals")
    fig_qq, ax = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("QQ Plot")
    st.pyplot(fig_qq)

# ------------------- Tab 4: Download + Try Yourself ---------------------
with tab4:
    st.subheader("📅 Download Predictions")
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residual': residuals})
    csv = results_df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="predictions.csv", mime="text/csv")

    st.subheader("🔍 Try Your Own Prediction")
    input_data = {}
    for feat in top_features:
        input_data[feat] = st.number_input(f"{feat}", value=float(df[feat].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = pipeline.predict(input_df)[0]
        st.success(f"📊 Predicted y: {pred:.2f}")

# ------------------- Tab 5: HTML View ---------------------
with tab5:
    st.subheader("📘 Detailed HTML Pipeline View")
    html_code = estimator_html_repr(pipeline)
    safe_html = f"""<iframe srcdoc=\"{html_code.replace('"', '&quot;')}\" width="100%" height="400" style="border:none;"></iframe>"""
    components.html(safe_html, height=420)

# ------------------- Footer ---------------------
st.markdown("""
<hr>
<b>Built by:</b> Raj | 📩 <a href='https://linkedin.com/in/yourname'>LinkedIn</a> | 🖠️ <a href='https://github.com/yourname'>GitHub</a>
""", unsafe_allow_html=True)


