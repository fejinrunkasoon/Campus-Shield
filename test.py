import joblib

# 检查特征列文件
try:
    feature_columns = joblib.load('feature_columns.pkl')
    print("=== feature_columns.pkl ===")
    print(f"类型: {type(feature_columns)}")
    if isinstance(feature_columns, list):
        print(f"特征数量: {len(feature_columns)}")
        print(f"特征列表: {feature_columns}")
    else:
        print(f"内容: {feature_columns}")
except Exception as e:
    print(f"加载feature_columns.pkl失败: {e}")

# 检查LightGBM模型
try:
    lgb_model = joblib.load('lightgbm_model.pkl')
    print("\n=== lightgbm_model.pkl ===")
    print(f"类型: {type(lgb_model)}")
    try:
        print(f"特征数量: {lgb_model.num_feature()}")
    except:
        pass
except Exception as e:
    print(f"加载lightgbm_model.pkl失败: {e}")

# 检查XGBoost模型
try:
    xgb_model = joblib.load('xgboost_model.pkl')
    print("\n=== xgboost_model.pkl ===")
    print(f"类型: {type(xgb_model)}")
    try:
        print(f"特征数量: {len(xgb_model.get_booster().feature_names)}")
        print(f"特征名: {xgb_model.get_booster().feature_names}")
    except:
        pass
except Exception as e:
    print(f"加载xgboost_model.pkl失败: {e}")

# 检查scaler文件
try:
    scaler = joblib.load('scaler(2).pkl')
    print("\n=== scaler(2).pkl ===")
    print(f"类型: {type(scaler)}")
    try:
        print(f"缩放均值: {scaler.mean_}")
        print(f"缩放标准差: {scaler.scale_}")
    except:
        pass
except Exception as e:
    print(f"加载scaler(2).pkl失败: {e}")