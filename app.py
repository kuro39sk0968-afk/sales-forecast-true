import streamlit as st
import pandas as pd
from forecast_engine import (
    generate_sample_data,
    preprocess_weekly_sales,
    create_features,
    train_and_evaluate,
    predict_future,
)

st.set_page_config(page_title="売上需要予測システム", layout="wide")

st.title("週次売上 需要予測ダッシュボード")
st.markdown("過去の売上実績データから、LightGBMを用いて向こう4週間の需要を予測します。")

st.sidebar.header("1. データソース選択")
uploaded_file = st.sidebar.file_uploader("売上CSVをアップロード（任意）", type=("csv",))

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        required_cols = {"date", "store_id", "item_id", "sales_quantity"}
        if not required_cols.issubset(raw_df.columns):
            st.error(f"CSVには以下の列が必要です: {required_cols}")
            st.stop()

        raw_df["sales_quantity"] = pd.to_numeric(raw_df["sales_quantity"], errors="coerce")
        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
        if raw_df.loc[:, ("date", "sales_quantity")].isnull().any().any():
            st.error("CSV内の日付または売上数量に不正な値（文字列・空欄など）が含まれています。")
            st.stop()
    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {e}")
        st.stop()
else:
    st.sidebar.info("CSV未アップロードのため、デモデータを使用しています。")
    raw_df = generate_sample_data()

st.sidebar.header("2. 予測対象の絞り込み")
store_ids = sorted(raw_df["store_id"].unique())
selected_store = st.sidebar.selectbox("店舗IDを選択", store_ids)

item_ids = sorted(raw_df[raw_df["store_id"] == selected_store]["item_id"].unique())
selected_item = st.sidebar.selectbox("商品IDを選択", item_ids)

weekly_df = preprocess_weekly_sales(raw_df, store_id=selected_store, item_id=selected_item)

if len(weekly_df) < 12:
    st.warning("学習に必要なデータ期間（最低12週間以上）が不足しています。")
    st.stop()

featured_df, features, targets = create_features(weekly_df)
models, metrics_df = train_and_evaluate(featured_df, features, targets)
future_df = predict_future(models, featured_df, features)

col1, col2 = st.columns((2, 1))

with col1:
    st.subheader(f"売上推移と4週間予測（店舗: {selected_store} / 商品: {selected_item}）")
    recent_actual = weekly_df.tail(24).rename(columns={"sales_quantity": "実績売上"})
    future_plot = future_df.loc[:, ("predicted_sales",)].rename(columns={"predicted_sales": "予測売上"})

    last_actual_point = pd.DataFrame(
        {"予測売上": (recent_actual["実績売上"].iloc[-1],)},
        index=(recent_actual.index[-1],),
    )
    connected_future = pd.concat((last_actual_point, future_plot))
    chart_df = pd.concat((recent_actual, connected_future), axis=1)
    st.line_chart(chart_df)

with col2:
    st.subheader("向こう4週間の予測値")
    display_future = future_df.reset_index()
    display_future["date"] = display_future["date"].dt.strftime("%Y-%m-%d")
    display_future.columns = ("対象週（日曜基準）", "予測期間", "予測売上数量")
    st.dataframe(display_future, use_container_width=True, hide_index=True)

    csv_data = display_future.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="予測結果をCSVでダウンロード",
        data=csv_data,
        file_name=f"forecast_store{selected_store}_item{selected_item}.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("モデルの検証精度（直近4週間のテストデータ評価）")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)
