import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("模型诊断报告")
print("=" * 60)

# 加载所有模型文件
xgb_model = joblib.load('xgboost_model.pkl')
lgb_model = joblib.load('lightgbm_model.pkl')
scaler = joblib.load('scaler(2).pkl')
feature_columns = joblib.load('feature_columns.pkl')

print("\n1. 特征列信息")
print(f"   特征数量: {len(feature_columns)}")
print(f"   特征列表: {feature_columns}")

print("\n2. XGBoost 模型信息")
print(f"   类型: {type(xgb_model)}")
print(f"   classes_: {xgb_model.classes_}")
print(f"   n_features_in_: {xgb_model.n_features_in_}")

print("\n3. LightGBM 模型信息")
print(f"   类型: {type(lgb_model)}")
print(f"   classes_: {lgb_model.classes_}")
print(f"   n_features_: {lgb_model.n_features_}")

print("\n4. Scaler 信息")
print(f"   类型: {type(scaler)}")
print(f"   mean_: {scaler.mean_}")
print(f"   scale_: {scaler.scale_}")
print(f"   n_features_in_: {scaler.n_features_in_}")
if hasattr(scaler, 'feature_names_in_'):
    print(f"   feature_names_in_: {scaler.feature_names_in_}")

print("\n5. 特征索引验证")
for i, name in enumerate(feature_columns):
    print(f"   索引 {i:2d}: {name}")

print("\n6. 测试预测一致性")
# 创建测试样本 - 正常交易（低金额）
features_normal = np.zeros((1, 30))
features_normal[0, 4] = 0.2   # V4 低频
features_normal[0, 10] = 0.7  # V10 正常波动
features_normal[0, 12] = 0.6  # V12 正常设备
features_normal[0, 14] = 0.8  # V14 正常位置
features_normal[0, 29] = 50   # Amount 低金额

# 创建测试样本 - 欺诈交易（高金额+异常）
features_fraud = np.zeros((1, 30))
features_fraud[0, 4] = 0.9    # V4 高频
features_fraud[0, 10] = -0.9  # V10 异常波动
features_fraud[0, 12] = -0.8  # V12 异常设备
features_fraud[0, 14] = -0.8  # V14 异常位置
features_fraud[0, 29] = 5000  # Amount 高金额

for name, features in [("正常交易", features_normal), ("欺诈交易", features_fraud)]:
    features_scaled = features.copy()
    features_scaled[:, [0, 29]] = scaler.transform(features[:, [0, 29]])
    
    xgb_proba = xgb_model.predict_proba(features_scaled)[0]
    lgb_proba = lgb_model.predict_proba(features_scaled)[0]
    
    print(f"\n   {name}:")
    print(f"   XGBoost proba: {xgb_proba} (class 0: {xgb_proba[0]:.4f}, class 1: {xgb_proba[1]:.4f})")
    print(f"   LightGBM proba: {lgb_proba} (class 0: {lgb_proba[0]:.4f}, class 1: {lgb_proba[1]:.4f})")
    
    # 判断哪个是欺诈类（应该是类别1）
    xgb_fraud_prob = xgb_proba[1]
    lgb_fraud_prob = lgb_proba[1]
    print(f"   → XGBoost 欺诈概率: {xgb_fraud_prob:.4f}")
    print(f"   → LightGBM 欺诈概率: {lgb_fraud_prob:.4f}")

print("\n7. 验证 scaler 的 feature_names_in_")
if hasattr(scaler, 'feature_names_in_'):
    print(f"   Scaler 训练时的特征名: {scaler.feature_names_in_}")
else:
    print("   Scaler 没有 feature_names_in_ 属性（可能是旧版sklearn）")

print("\n8. 验证特征顺序")
print(f"   当前代码使用的特征顺序: {feature_columns}")
print(f"   V4 索引: {feature_columns.index('V4')}")
print(f"   V10 索引: {feature_columns.index('V10')}")
print(f"   V12 索引: {feature_columns.index('V12')}")
print(f"   V14 索引: {feature_columns.index('V14')}")
print(f"   Amount 索引: {feature_columns.index('Amount')}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)