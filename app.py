import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from supabase import create_client, Client
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
    client = genai.Client()  # GEMINI_API_KEY または GOOGLE_API_KEY を自動取得
except Exception:
    st.error("Google APIキーが設定されていません。環境変数 GEMINI_API_KEY を設定してください。")
    st.stop()

@st.cache_resource
def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        st.error("Supabaseの環境変数（SUPABASE_URL / SUPABASE_KEY）が設定されていません。")
        st.stop()
    return create_client(url, key)

supabase = get_supabase_client()

# --- 2. ユーティリティ関数 ---

def upload_to_storage(uploaded_file) -> tuple[str | None, str | None]:
    """UploadedFile を Supabase Storage にアップロードし (public_url, mime_type) を返す"""
    if uploaded_file is None:
        return None, None
    ext = uploaded_file.name.split('.')[-1]
    file_path = f"{uuid.uuid4()}.{ext}"
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    mime_type = mime_type or "application/octet-stream"
    byte_data = uploaded_file.getvalue()
    supabase.storage.from_("media").upload(
        file_path, byte_data, file_options={"content-type": mime_type}
    )
    public_url = supabase.storage.from_("media").get_public_url(file_path)
    return public_url, mime_type


def normalize_vector(vec):
    """768次元のL2正規化（検索精度を保つため必須）"""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


# --- 3. 埋め込み関数（メモリ上でバイト処理）---
def create_embedding(
    text_memo: str,
    image_files=None,
    audio_file=None,
    video_file=None,
    pdf_file=None,
):
    """UploadedFile / bytes をメモリ上で直接 Gemini Embedding API に渡す"""
    contents = []

    if text_memo:
        contents.append(text_memo)

    for f in (image_files or []):
        if f is not None:
            f.seek(0)
            byte_data = f.read()
            mime_type, _ = mimetypes.guess_type(f.name)
            contents.append(types.Part.from_bytes(data=byte_data, mime_type=mime_type or "image/jpeg"))

    for f in [audio_file, video_file, pdf_file]:
        if f is not None:
            f.seek(0)
            byte_data = f.read()
            mime_type, _ = mimetypes.guess_type(f.name)
            if not mime_type and f.name.lower().endswith('.pdf'):
                mime_type = "application/pdf"
            contents.append(types.Part.from_bytes(data=byte_data, mime_type=mime_type or "application/octet-stream"))

    if not contents:
        return None

    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        embedding = result.embeddings[0].values
        return normalize_vector(embedding)
    except APIError as e:
        st.error(f"⚠️ APIエラー（ファイル上限超過の可能性）: {e}")
        return None
    except Exception as e:
        st.error(f"⚠️ 予期せぬエラー: {e}")
        return None


# --- 4. Streamlit UI ---
st.set_page_config(page_title="TripRecall v2.0", page_icon="🗺️", layout="centered")
st.title("🗺️ TripRecall v2.0 - 旅の思い出を呼び起こす")
st.markdown("テキスト、写真(最大6枚)、音声、動画、PDFを**1つの記憶（ベクトル）**に統合。ふとしたキーワードから、旅行の記憶をタイムラインで蘇らせます。")

tab1, tab2 = st.tabs(["📸 新しい思い出を登録", "🔍 思い出を呼び出す"])

# ============== タブ1: 思い出の登録 ==============
with tab1:
    st.subheader("旅行中の1シーンを登録")

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            default_date = st.session_state.get("exif_date", datetime.today())
            date = st.date_input("🗓️ 日付", default_date)
        with col_b:
            location = st.text_input("📍 場所（例：京都・清水寺）", "京都・清水寺")

        text_memo = st.text_area("📝 テキストメモ（任意）", st.session_state.get("ai_memo", "雨の日の静かな寺院..."))

        st.markdown("##### 📎 記憶のメディアを追加（組み合わせ自由）")

        image_files = st.file_uploader("写真🖼️（任意・最大6枚で空気感を演出）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # アイデアB: EXIFから撮影日時を自動入力
        if image_files:
            try:
                first_img = Image.open(image_files[0])
                exif_data = first_img._getexif()
                if exif_data:
                    tag_map = {v: k for k, v in ExifTags.TAGS.items()}
                    date_tag = tag_map.get("DateTimeOriginal")
                    if date_tag and date_tag in exif_data:
                        dt_str = exif_data[date_tag]
                        exif_dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                        if st.session_state.get("exif_date") != exif_dt.date():
                            st.session_state["exif_date"] = exif_dt.date()
                            st.rerun()
            except Exception:
                pass

        # アイデアA: AIによるテキストメモ自動生成
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
            if image_files and len(image_files) > 6:
                st.warning("⚠️ 写真は最大6枚までです！最初の6枚のみを保存します。")
                image_files = image_files[:6]

            if not any([text_memo, image_files, audio_file, video_file, pdf_file]):
                st.error("⚠️ いずれかの思い出データを入力してください！")
            else:
                with st.spinner("メディアをクラウドに保存して思い出ベクトルを生成中..."):
                    # 埋め込みはアップロード前にメモリ上で生成（ファイルポインタをリセット）
                    for f in (image_files or []):
                        f.seek(0)
                    if audio_file: audio_file.seek(0)
                    if video_file: video_file.seek(0)
                    if pdf_file: pdf_file.seek(0)

                    embedding = create_embedding(text_memo, image_files, audio_file, video_file, pdf_file)

                    if embedding:
                        # Storage へアップロード（ファイルポインタを再リセット）
                        img_urls = []
                        for img in (image_files or []):
                            img.seek(0)
                            url, _ = upload_to_storage(img)
                            if url:
                                img_urls.append(url)

                        if audio_file: audio_file.seek(0)
                        audio_url, _ = upload_to_storage(audio_file)

                        if video_file: video_file.seek(0)
                        video_url, _ = upload_to_storage(video_file)

                        if pdf_file: pdf_file.seek(0)
                        pdf_url, _ = upload_to_storage(pdf_file)

                        supabase.table("trip_memories").insert({
                            "date": str(date),
                            "location": location,
                            "text_memo": text_memo,
                            "image_paths": img_urls,  # listをそのまま渡す（JSONB自動変換）
                            "audio_path": audio_url or "",
                            "video_path": video_url or "",
                            "pdf_path": pdf_url or "",
                            "timestamp": datetime.now().isoformat(),
                            "embedding": embedding,
                        }).execute()

                        st.success("✅ 思い出をクラウドに保存しました！「思い出を呼び出す」タブで検索してみましょう。")

# ============== タブ2: 思い出の呼び出し ==============
with tab2:
    st.subheader("思い出の糸をたぐり寄せる")
    st.write("言葉や手元の写真から、過去の思い出を「タイムライン」として蘇らせます。")

    query_text = st.text_input("🔍 検索キーワード", "京都の雨の日の寺院の雰囲気")
    query_image = st.file_uploader("🖼️ 参考写真で検索（任意）", type=["jpg", "jpeg", "png"])
    threshold = st.slider("🎯 検索の一致度（高いほど厳密・低いほどふんわり）", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

    if st.button("検索してタイムライン再生", type="primary", use_container_width=True):
        with st.spinner("記憶を検索中..."):
            query_embedding = create_embedding(
                query_text,
                image_files=[query_image] if query_image else []
            )

            if query_embedding:
                response = supabase.rpc("match_memories", {
                    "query_embedding": query_embedding,
                    "match_threshold": threshold,
                    "match_count": 5
                }).execute()
                results = response.data

                if not results:
                    st.info("指定した条件に合う思い出は見つかりませんでした。一致度を下げてみてください。")
                else:
                    st.write("### 🎬 思い出のタイムライン")
                    # 日付・タイムスタンプ順にソート
                    results.sort(key=lambda x: (x.get("date", ""), x.get("timestamp", "")))

                    for meta in results:
                        similarity = meta.get("similarity", 0)
                        with st.container(border=True):
                            st.markdown(f"#### 🗓️ {meta.get('date', '')} 📍 {meta.get('location', '')}")
                            st.caption(f"AI類似度: {similarity:.3f} (1に近いほど検索意図に一致)")

                            # 複数画像のグリッド表示（str/list 混在データに対応）
                            raw_paths = meta.get("image_paths", [])
                            if isinstance(raw_paths, str):
                                try:
                                    img_urls = json.loads(raw_paths)
                                except Exception:
                                    img_urls = []
                            else:
                                img_urls = raw_paths or []
                            if img_urls:
                                cols = st.columns(min(len(img_urls), 3))
                                for idx, url in enumerate(img_urls):
                                    cols[idx % 3].image(url, use_container_width=True)

                            media_cols = st.columns(2)
                            with media_cols[0]:
                                video_url = meta.get("video_path", "")
                                if video_url:
                                    st.video(video_url)
                            with media_cols[1]:
                                audio_url = meta.get("audio_path", "")
                                if audio_url:
                                    st.audio(audio_url)

                            pdf_url = meta.get("pdf_path", "")
                            if pdf_url:
                                st.markdown(f"[📄 添付されたPDFを開く]({pdf_url})")

                            if meta.get("text_memo"):
                                st.info(f"📝 {meta['text_memo']}")
