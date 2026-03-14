# TripRecall

TripRecall は、旅行の思い出を記録し、キーワードや画像からあの時の記憶をタイムラインとして鮮やかに蘇らせるアプリケーションです。
テキスト、写真、音声、動画、PDF などを1つの「記憶（ベクトル）」に統合し、Gemini の埋め込みモデルと ChromaDB を活用して高度な類似検索を実現します。

## 主な機能

- **思い出の登録**:
  - 日付、場所、テキストメモ
  - 写真（最大6枚）、PDF（最大6ページ）、音声（最大80秒）、動画（最大128秒）を組み合わせて登録
  - 全てのメディアをマルチモーダルなベクトルデータに変換し一元管理
- **思い出の呼び出し**:
  - キーワード（例: 「京都の雨の日の寺院の雰囲気」）や参考画像を使った柔軟な類似検索
  - 検索結果を日付順（タイムライン）で表示し、当時のメディアと一緒に旅行の記憶を振り返ることができます

## 使用技術

- [Streamlit](https://streamlit.io/) - Web UIフレームワーク
- [Google GenAI (Gemini)](https://ai.google.dev/) - 埋め込みモデル (`gemini-embedding-2-preview`) 
- [ChromaDB](https://www.trychroma.com/) - ベクトルデータベース
- Python 3

## 環境構築と実行方法

1. リポジトリのクローン
   ```bash
   git clone <REPOSITORY_URL>
   cd TripRecall
   ```

2. 必要なパッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```

3. 環境変数の設定
   プロジェクトのルートディレクトリに `.env` ファイルを作成するか、環境変数として `GEMINI_API_KEY` を設定してください。
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

4. アプリケーションの起動
   ```bash
   streamlit run app.py
   ```

## 注意事項

- メディア登録時にAPIの制限（動画128秒、音声80秒、PDF6ページ、画像6枚まで）を超える場合はエラーとなるか、切り捨てられます。
- 初期起動時に ChromaDB ファイル等の保存用ディレクトリ (`./media`, `./chroma_db`) が自動的に作成されます。
