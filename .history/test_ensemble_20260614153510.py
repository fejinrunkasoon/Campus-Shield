import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("双模型分歧分析")
print("=" * 70)

# 加载模型
xgb_model = joblib.load('xgboost_model.pkl')
lgb_model = joblib.load('lightgbm_model.pkl')
scaler = joblib.load('scaler(2).pkl')
feature_columns = joblib.load('feature_columns.pkl')

# 生成测试样本（模拟真实数据分布）
np.random.seed(42)
n_samples = 1000

# 生成特征
time = np.random.uniform(0, 172800, n_samples)  # 2天内
v1_v28 = np.random.randn(n_samples, 28) * 0.5  # 标准正态分布
amount = np.random.lognormal(4, 1.5, n_samples)  # 对数正态分布

# 组装特征
X = np.zeros((n_samples, 30))
X[:, 0] = time
X[:, 1:29] = v1_v28
X[:, 29] = amount

# 缩放 Time 和 Amount
X_scaled = X.copy()
X_scaled[:, [0, 29]] = scaler.transform(X[:, [0, 29]])

# 预测
xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
lgb_proba = lgb_model.predict_proba(X_scaled)[:, 1]

# 计算差异
diff = np.abs(xgb_proba - lgb_proba)
ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba

print(f"\n1. 整体统计")
print(f"   样本总数: {n_samples}")
print(f"   XGBoost 平均概率: {xgb_proba.mean():.4f}")
print(f"   LightGBM 平均概率: {lgb_proba.mean():.4f}")
print(f"   平均差异: {diff.mean():.4f}")
print(f"   最大差异: {diff.max():.4f}")
print(f"   差异标准差: {diff.std():.4f}")

print(f"\n2. 分歧样本分析")
thresholds = [0.1, 0.2, 0.3, 0.5]
for thresh in thresholds:
    count = np.sum(diff > thresh)
    pct = count / n_samples * 100
    print(f"   差异 > {thresh:.1f}: {count} 样本 ({pct:.2f}%)")

print(f"\n3. 极端分歧样本检查 (差异 > 0.3)")
extreme_mask = diff > 0.3
if np.any(extreme_mask):
    extreme_idx = np.where(extreme_mask)[0][:5]  # 前5个
    for idx in extreme_idx:
        print(f"   样本 {idx}:")
        print(f"     XGBoost: {xgb_proba[idx]:.4f} ({xgb_proba[idx]*100:.2f}%)")
        print(f"     LightGBM: {lgb_proba[idx]:.4f} ({lgb_proba[idx]*100:.2f}%)")
        print(f"     差异: {diff[idx]:.4f}")
        print(f"     集成: {ensemble_proba[idx]:.4f}")
        # 显示关键特征
        print(f"     Time: {X[idx, 0]:.1f}, Amount: {X[idx, 29]:.2f}")
        print(f"     V4: {X[idx, 4]:.3f}, V10: {X[idx, 10]:.3f}, V12: {X[idx, 12]:.3f}, V14: {X[idx, 14]:.3f}")
else:
    print("   未发现极端分歧样本")

print(f"\n4. 决策一致性检查（以 0.5 为阈值）")
xgb_pred = (xgb_proba > 0.5).astype(int)
lgb_pred = (lgb_proba > 0.5).astype(int)
ensemble_pred = (ensemble_proba > 0.5).astype(int)

agreement = np.sum(xgb_pred == lgb_pred) / n_samples * 100
print(f"   两模型决策一致率: {agreement:.2f}%")

# 检查分歧样本的决策
 disagreement_mask = xgb_pred != lgb_pred
if np.any(disagreement_mask):
    disagree_count = np.sum(disagreement_mask)
    print(f"   决策分歧样本数: {disagree_count} ({disagree_count/n_samples*100:.2f}%)")
    
    # 分析分歧类型
    xgb1_lgb0 = np.sum((xgb_pred == 1) & (lgb_pred == 0))
    xgb0_lgb1 = np.sum((xgb_pred == 0) & (lgb_pred == 1))
    print(f"   XGBoost判欺诈, LightGBM判正常: {xgb1_lgb0}")
    print(f"   XGBoost判正常, LightGBM判欺诈: {xgb0_lgb1}")

print(f"\n5. 相关系数")
corr = np.corrcoef(xgb_proba, lgb_proba)[0, 1]
print(f"   Pearson 相关系数: {corr:.4f}")

print(f"\n6. Streamlit 场景复现测试")
# 模拟 Streamlit 中的三种场景
scenarios = [
    ("正常校园生活", 0.8, 0.2, 0.7, 0.6, 50),
    ("深夜异常高额消费", 0.3, 0.9, 0.2, 0.4, 500),
    ("疑似异地盗刷", -0.8, 0.1, -0.9, -0.8, 2000),
]

for name, geo, freq, fluct, device, amt in scenarios:
    features = np.zeros((1, 30))
    features[0, 4] = freq      # V4
    features[0, 10] = fluct    # V10
    features[0, 12] = device   # V12
    features[0, 14] = geo      # V14
    features[0, 29] = amt      # Amount
    
    features_scaled = features.copy()
    features_scaled[:, [0, 29]] = scaler.transform(features[:, [0, 29]])
    
    xgb_p = xgb_model.predict_proba(features_scaled)[0][1]
    lgb_p = lgb_model.predict_proba(features_scaled)[0][1]
    diff_p = abs(xgb_p - lgb_p)
    
    print(f"\n   {name}:")
    print(f"     XGBoost: {xgb_p:.4f} | LightGBM: {lgb_p:.4f} | 差异: {diff_p:.4f}")
    if diff_p > 0.2:
        print(f"     ⚠️ 差异超过 0.2")
    else:
        print(f"     ✅ 差异在可接受范围")

print("\n" + "=" * 70)