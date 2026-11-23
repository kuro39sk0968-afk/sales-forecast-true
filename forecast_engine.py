import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def generate_sample_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=300, freq="D")
    records = []
    for store in ("S01", "S02"):
        for item in ("ITEM_A", "ITEM_B"):
            base = 50 if store == "S01" else 30
            trend = np.linspace(0, 15, len(dates))
            seasonal = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
            noise = np.random.normal(0, 5, len(dates))
            sales = np.maximum(0, base + trend + seasonal + noise).astype(int)
            for d, s in zip(dates, sales):
                records.append({"date": d, "store_id": store, "item_id": item, "sales_quantity": s})
    return pd.DataFrame(records)


def preprocess_weekly_sales(df, store_id, item_id):
    sub = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    
    if sub.empty:
        return pd.DataFrame(columns=["sales_quantity"])

    # 日次で同日複数レコードがある場合に備えて日次合算し、欠損日を0埋め
    daily = sub.groupby("date")["sales_quantity"].sum().asfreq("D", fill_value=0)
    last_date = daily.index.max()

    # 日曜締め週次集計
    weekly = daily.resample("W-SUN").sum().to_frame()

    # 最終日が日曜日(6)でない場合、途中の不完全な週を除外して予測の過小評価を防ぐ
    if last_date.dayofweek != 6 and len(weekly) > 1:
        weekly = weekly.iloc[:-1]

    return weekly


def create_features(weekly_df):
    df = weekly_df.copy()
    for lag in (1, 2, 3, 4):
        df[f"lag_{lag}"] = df["sales_quantity"].shift(lag)
    
    # 直近4週間の移動平均と標準偏差（トレンドとブレ幅を捉える）
    df["rolling_mean_4"] = df["sales_quantity"].shift(1).rolling(4).mean()
    df["rolling_std_4"] = df["sales_quantity"].shift(1).rolling(4).std().fillna(0)
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    targets = []
    for h in range(1, 5):
        col = f"target_h{h}"
        df[col] = df["sales_quantity"].shift(-h)
        targets.append(col)

    features = [
        c for c in df.columns
        if c.startswith("lag_") or c.startswith("rolling_") or c == "week_of_year"
    ]
    return df, features, targets


def _build_lgb_model():
    # データ件数が少なくても分岐・学習できるようにハイパーパラメータを調整
    return lgb.LGBMRegressor(
        n_estimators=60,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=3,
        subsample_for_bin=50,
        random_state=42,
        verbose=-1,
    )


def train_and_evaluate(featured_df, features, targets):
    models = {}
    metrics = []
    train_data = featured_df.dropna(subset=features + targets)

    # データが極端に少ない場合の安全装置（新商品など）
    if len(train_data) < 6:
        fallback_val = float(featured_df["sales_quantity"].tail(4).mean()) if not featured_df.empty else 0.0
        for i, target in enumerate(targets, 1):
            models[target] = fallback_val  # 数値そのものをフォールバックモデルとして保持
            metrics.append({"予測期間": f"{i}週間先", "MAE（平均絶対誤差）": 0.0, "RMSE": 0.0})
        return models, pd.DataFrame(metrics)

    # 1. 精度検証用（直近4週をテストデータに分割）
    split_idx = max(1, len(train_data) - 4)
    train_df = train_data.iloc[:split_idx]
    val_df = train_data.iloc[split_idx:]

    for i, target in enumerate(targets, 1):
        eval_model = _build_lgb_model()
        eval_model.fit(train_df[features], train_df[target])

        if len(val_df) > 0:
            preds = np.maximum(0, eval_model.predict(val_df[features]))
            mae = mean_absolute_error(val_df[target], preds)
            rmse = root_mean_squared_error(val_df[target], preds)
        else:
            mae, rmse = 0.0, 0.0

        metrics.append({
            "予測期間": f"{i}週間先",
            "MAE（平均絶対誤差）": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

        # 2. 本番予測用（直近4週の最新トレンドも含めた全データで再学習）
        prod_model = _build_lgb_model()
        prod_model.fit(train_data[features], train_data[target])
        models[target] = prod_model

    return models, pd.DataFrame(metrics)


def predict_future(models, featured_df, features):
    if featured_df.empty:
        return pd.DataFrame(columns=["horizon", "predicted_sales"])

    latest_row = featured_df.iloc[-1:][features]
    last_date = featured_df.index[-1]
    future_records = []

    for i in range(1, 5):
        target = f"target_h{i}"
        model = models[target]

        # フォールバック（数値）か学習済みモデルかを判定して予測
        if isinstance(model, (int, float)):
            pred = max(0.0, float(model))
        else:
            pred = max(0.0, float(model.predict(latest_row).item()))

        future_date = last_date + pd.Timedelta(weeks=i)
        future_records.append({
            "date": future_date,
            "horizon": f"{i}週間先",
            "predicted_sales": round(pred, 1)
        })

    return pd.DataFrame(future_records).set_index("date")
