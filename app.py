import streamlit as st
import chromadb
from google import genai
from google.genai import types
from google.genai.errors import APIError
import os
from datetime import datetime
import uuid
import mimetypes
import numpy as np
import json
from dotenv import load_dotenv
from PIL import Image, ExifTags

# .envファイルの読み込み
load_dotenv()

# --- 1. 初期設定 ---
try:
    client = genai.Client() # GEMINI_API_KEY または GOOGLE_API_KEY を自動取得
except Exception as e:
    st.error("Google APIキーが設定されていません。環境変数 GEMINI_API_KEY を設定してください。")
    st.stop()

# v1とデータ構造が変わるためコレクション名を新しくします
COLLECTION_NAME = "trip_memories_v1_1"
MEDIA_DIR = "./media"
DB_DIR = "./chroma_db"
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_DIR)

client_chroma = get_chroma_client()
collection = client_chroma.get_or_create_collection(COLLECTION_NAME)

# --- 2. ユーティリティ関数 ---
def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    ext = uploaded_file.name.split('.')[-1]
    file_path = os.path.join(MEDIA_DIR, f"{uuid.uuid4()}.{ext}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ★ChatGPTレビュー反映: 768次元のL2正規化（検索精度を保つため必須）
def normalize_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()

# --- 3. 埋め込み関数 ---
def create_embedding(text_memo: str, image_paths=[], audio_path=None, video_path=None, pdf_path=None):
    contents = []
    
    if text_memo:
        contents.append(text_memo)
        
    # ★ChatGPTレビュー反映: 複数画像（最大6枚）の処理
    for path in image_paths:
        if path and os.path.exists(path):
            mime_type, _ = mimetypes.guess_type(path)
            with open(path, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type or "image/jpeg"))

    # 音声・動画・PDFの処理
    media_paths = [audio_path, video_path, pdf_path]
    for path in media_paths:
        if path and os.path.exists(path):
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type and path.lower().endswith('.pdf'):
                mime_type = "application/pdf"
            with open(path, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type or "application/octet-stream"))
                
    if not contents:
        return None
        
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        embedding = result.embeddings[0].values
        return normalize_vector(embedding) # 正規化して返す
    except APIError as e:
        # ★ChatGPTレビュー反映: API側の制限エラー（秒数やページ数超過など）をキャッチ
        st.error(f"⚠️ APIエラーが発生しました。ファイルの上限（動画128秒、音声80秒、PDF6ページ、画像6枚）を超過している可能性があります。\n詳細: {e}")
        return None
    except Exception as e:
        st.error(f"⚠️ 予期せぬエラーが発生しました: {e}")
        return None

# --- 4. Streamlit UI ---
st.set_page_config(page_title="TripRecall v1.1", page_icon="🗺️", layout="centered")
st.title("🗺️ TripRecall v1.1 - 旅の思い出を呼び起こす")
st.markdown("テキスト、写真(最大6枚)、音声、動画、PDFを**1つの記憶（ベクトル）**に統合。ふとしたキーワードから、旅行の記憶をタイムラインで蘇らせます。")

tab1, tab2 = st.tabs(["📸 新しい思い出を登録", "🔍 思い出を呼び出す"])

# ============== タブ1: 思い出の登録 ==============
with tab1:
    st.subheader("旅行中の1シーンを登録")
    
    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            # アイデアB: EXIFで取得した日付があればそれを初期値に使用
            default_date = st.session_state.get("exif_date", datetime.today())
            date = st.date_input("🗓️ 日付", default_date)
        with col_b:
            location = st.text_input("📍 場所（例：京都・清水寺）", "京都・清水寺")

        # アイデアA: AIが生成したメモがあればそれを初期値に使用
        text_memo = st.text_area("📝 テキストメモ（任意）", st.session_state.get("ai_memo", "雨の日の静かな寺院..."))
        
        st.markdown("##### 📎 記憶のメディアを追加（組み合わせ自由）")
        
        # accept_multiple_files=True で複数選択可能に
        image_files = st.file_uploader("写真🖼️（任意・最大6枚で空気感を演出）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # アイデアB: 写真がアップロードされたらEXIFを解析して日付を自動入力
        if image_files:
            try:
                first_img = Image.open(image_files[0])
                exif_data = first_img._getexif()
                if exif_data:
                    tag_map = {v: k for k, v in ExifTags.TAGS.items()}
                    date_tag = tag_map.get("DateTimeOriginal")
                    if date_tag and date_tag in exif_data:
                        dt_str = exif_data[date_tag]  # 例: '2024:07:15 10:30:00'
                        exif_dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                        if st.session_state.get("exif_date") != exif_dt.date():
                            st.session_state["exif_date"] = exif_dt.date()
                            st.rerun()
            except Exception:
                pass  # EXIFが読めない場合は何もしない

        # アイデアA: AIによるテキストメモ自動生成ボタン
        if st.button("✨ 写真からAIに思い出を綴ってもらう", disabled=not image_files):
            with st.spinner("AIが情景を描写中..."):
                try:
                    first_file = image_files[0]
                    first_file.seek(0)
                    img_bytes = first_file.read()
                    mime_type, _ = mimetypes.guess_type(first_file.name)
                    mime_type = mime_type or "image/jpeg"
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                            "この旅行写真の情景を、思い出に残るような少しエモーショナルな文章（150文字程度）で描写してください。"
                        ]
                    )
                    st.session_state["ai_memo"] = response.text.strip()
                    st.rerun()
                except Exception as e:
                    st.error(f"AIテキスト生成に失敗しました: {e}")
        pdf_file = st.file_uploader("PDF📄（任意・旅程表やパンフなど 最大6ページ）", type=["pdf"])

        col1, col2 = st.columns(2)
        with col1:
            audio_file = st.file_uploader("現地音声🎙️（任意・最大80秒）", type=["mp3", "wav", "m4a"])
        with col2:
            video_file = st.file_uploader("動画🎥（任意・最大128秒）", type=["mp4", "mov"])
        
        if st.button("✨ このシーンをAI空間に保存", type="primary", use_container_width=True):
            # 事前バリデーション
            if image_files and len(image_files) > 6:
                st.warning("⚠️ 写真は最大6枚までです！最初の6枚のみを保存します。")
                image_files = image_files[:6]
                
            if not any([text_memo, image_files, audio_file, video_file, pdf_file]):
                st.error("⚠️ いずれかの思い出データ（テキスト、写真、音声、動画、PDF）を入力してください！")
            else:
                with st.spinner("メディアを統合して思い出ベクトルを生成中..."):
                    img_paths = [save_uploaded_file(img) for img in image_files] if image_files else []
                    audio_path = save_uploaded_file(audio_file)
                    video_path = save_uploaded_file(video_file)
                    pdf_path = save_uploaded_file(pdf_file)
                    
                    embedding = create_embedding(text_memo, img_paths, audio_path, video_path, pdf_path)
                    
                    if embedding:
                        doc_id = str(uuid.uuid4())
                        collection.add(
                            ids=[doc_id],
                            embeddings=[embedding],
                            metadatas=[{
                                "date": str(date),
                                "location": location,
                                "text": text_memo,
                                # ChromaDBはリストを直接保存できないためJSON文字列化して保存
                                "image_paths": json.dumps(img_paths), 
                                "audio_path": audio_path or "",
                                "video_path": video_path or "",
                                "pdf_path": pdf_path or "",
                                "timestamp": datetime.now().isoformat()
                            }]
                        )
                        st.success("✅ 思い出を保存しました！「思い出を呼び出す」タブで検索してみましょう。")

# ============== タブ2: 思い出の呼び出し ==============
with tab2:
    st.subheader("思い出の糸をたぐり寄せる")
    st.write("言葉や手元の写真から、過去の思い出を「タイムライン」として蘇らせます。")
    
    query_text = st.text_input("🔍 検索キーワード", "京都の雨の日の寺院の雰囲気")
    query_image = st.file_uploader("🖼️ 参考写真で検索（任意）", type=["jpg", "jpeg", "png"])
    
    if st.button("検索してタイムライン再生", type="primary", use_container_width=True):
        with st.spinner("記憶を検索中..."):
            query_img_path = save_uploaded_file(query_image)
            
            query_embedding = create_embedding(query_text, image_paths=[query_img_path] if query_img_path else [])
            
            if query_embedding:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5, # 類似度トップ5を取得
                    include=["metadatas", "distances"]
                )
                
                if not results["metadatas"] or not results["metadatas"][0]:
                    st.info("思い出が見つかりませんでした。")
                else:
                    st.write("### 🎬 思い出のタイムライン")
                    
                    metadatas = results["metadatas"][0]
                    distances = results["distances"][0]
                    
                    combined = []
                    for m, d in zip(metadatas, distances):
                        m["_distance"] = d
                        combined.append(m)
                        
                    # ★ChatGPTレビュー反映: 類似検索後、日付(date)と登録時間の古い順にソート（タイムライン化）
                    combined.sort(key=lambda x: (x["date"], x["timestamp"]))
                    
                    for meta in combined:
                        distance = meta["_distance"]
                        with st.container(border=True):
                            st.markdown(f"#### 🗓️ {meta['date']} 📍 {meta['location']}")
                            st.caption(f"AI類似度距離: {distance:.3f} (0に近いほど検索意図に一致)")
                            
                            # 複数画像の復元とグリッド表示
                            img_paths = json.loads(meta.get('image_paths', '[]'))
                            if img_paths:
                                valid_paths = [p for p in img_paths if os.path.exists(p)]
                                if valid_paths:
                                    cols = st.columns(min(len(valid_paths), 3)) # 最大3列で折り返し
                                    for idx, p in enumerate(valid_paths):
                                        cols[idx % 3].image(p, use_container_width=True)
                            
                            media_cols = st.columns(2)
                            with media_cols[0]:
                                if meta.get('video_path') and os.path.exists(meta['video_path']):
                                    st.video(meta['video_path'])
                            with media_cols[1]:
                                if meta.get('audio_path') and os.path.exists(meta['audio_path']):
                                    st.audio(meta['audio_path'])
                                    
                            if meta.get('pdf_path') and os.path.exists(meta['pdf_path']):
                                with open(meta['pdf_path'], "rb") as f:
                                    st.download_button(
                                        label=f"📄 添付されたPDFを開く",
                                        data=f.read(),
                                        file_name=f"document_{meta['date']}.pdf",
                                        mime="application/pdf",
                                        key=f"pdf_{meta['timestamp']}"
                                    )
                                    
                            if meta.get('text'):
                                st.info(f"📝 {meta['text']}")
                                
            # 検索用の一時画像を削除してお掃除
            if query_img_path and os.path.exists(query_img_path):
                os.remove(query_img_path)
