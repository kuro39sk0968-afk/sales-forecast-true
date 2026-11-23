# 週次売上 需要予測ダッシュボード（Webアプリ）

LightGBMを用いた向こう4週間の需要予測Webアプリケーションです。企画・要件定義からPythonでの実装までを一貫して開発しました。

<!-- 公開URLや画像があればここに配置 -->
<!-- 🔗 **デモアプリ:** [URLをここに貼る] -->
<!-- ![ダッシュボード画面](images/screenshot.png) -->

## 使用技術・ライブラリ
- **言語:** Python 3.x
- **Webフレームワーク:** Streamlit
- **機械学習・データ処理:** LightGBM, scikit-learn, pandas, NumPy
- **設計・資料:** 要件定義書、画面設計書（PDF格納）

## システム構成・工夫した点
- **モジュール分割:** 予測ロジック（`forecast_engine.py`）とUI（`app.py`）を分離し、保守性を向上。
- **時系列予測:** ラグ特徴量や移動平均を生成し、データリークを防いだ検証を実施。
- **実務的なUI/UX:** デモデータ自動生成機能や結果のCSVダウンロード機能を実装。

## セットアップ・起動方法
```bash
# 1. 依存ライブラリのインストール
pip install -r requirements.txt

# 2. アプリの起動
streamlit run app.py
```

## 格納ファイル一覧
- `app.py` : Streamlitを用いたダッシュボード画面
- `forecast_engine.py` : LightGBM学習および予測エンジン
- `requirements.txt` : 依存パッケージ一覧
- `1_需要予測Webアプリ概要書_tamori.pdf` 等 : 企画・画面設計資料
