import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 加载模型
xgb_model = joblib.load('xgboost_model.pkl')
lgb_model = joblib.load('lightgbm_model.pkl')
scaler = joblib.load('scaler(2).pkl')
feature_columns = joblib.load('feature_columns.pkl')

def create_feature_vector_fixed(geo_deviation, txn_frequency, acc_fluctuation, device_risk, amount):
    base_features = np.zeros(30)
    base_features[4] = txn_frequency    # V4
    base_features[10] = acc_fluctuation # V10
    base_features[12] = device_risk     # V12
    base_features[14] = geo_deviation   # V14
    base_features[29] = amount          # Amount
    return base_features.reshape(1, -1)

print("=" * 60)
print("修复验证：测试三种场景")
print("=" * 60)

scenarios = [
    ("正常校园生活", 0.8, 0.2, 0.7, 0.6, 50),
    ("深夜异常高额消费", 0.3, 0.9, 0.2, 0.4, 500),
    ("疑似异地盗刷", -0.8, 0.1, -0.9, -0.8, 2000),
]

for name, geo, freq, fluct, device, amt in scenarios:
    features = create_feature_vector_fixed(geo, freq, fluct, device, amt)
    features_scaled = features.copy()
    features_scaled[:, [0, 29]] = scaler.transform(features[:, [0, 29]])
    
    xgb_proba = xgb_model.predict_proba(features_scaled)[0][1]
    lgb_proba = lgb_model.predict_proba(features_scaled)[0][1]
    ensemble = 0.5 * xgb_proba + 0.5 * lgb_proba
    
    print(f"\n场景: {name}")
    print(f"  参数: geo={geo}, freq={freq}, fluct={fluct}, device={device}, amount={amt}")
    print(f"  XGBoost: {xgb_proba:.4f} ({xgb_proba*100:.2f}%)")
    print(f"  LightGBM: {lgb_proba:.4f} ({lgb_proba*100:.2f}%)")
    print(f"  集成: {ensemble:.4f} ({ensemble*100:.2f}%)")
    
    # 判断是否一致
    diff = abs(xgb_proba - lgb_proba)
    if diff < 0.3:
        print(f"  ✅ 两模型一致 (差异: {diff:.4f})")
    else:
        print(f"  ⚠️ 两模型差异较大 (差异: {diff:.4f})")

print("\n" + "=" * 60)