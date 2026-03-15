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
    client = genai.Client()
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
        return normalize_vector(result.embeddings[0].values)
    except APIError as e:
        st.error(f"⚠️ APIエラー（ファイル上限超過の可能性）: {e}")
        return None
    except Exception as e:
        st.error(f"⚠️ 予期せぬエラー: {e}")
        return None


# --- 4. Streamlit UI ---
st.set_page_config(page_title="MomentWeave", page_icon="📖", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@400;500;700&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Zen Maru Gothic', sans-serif !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 24px rgba(60, 49, 42, 0.06) !important;
        border: 1px solid #EDE4D9 !important;
        padding: 1.5rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(60, 49, 42, 0.1) !important;
    }

    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #FF9A5C, #E36F3D) !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(255, 154, 92, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 154, 92, 0.4) !important;
    }

    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #FF9A5C !important;
        border-radius: 16px !important;
        background-color: #FFF9F0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📖 MomentWeave")
st.markdown("毎日のささやかな瞬間、感情、空気感を優しく織りなす、あなただけの記憶のノートです。")

tab1, tab2 = st.tabs(["🖋️ 今日のページを綴る", "🔍 記憶の糸をたぐる"])

# ============== タブ1: 今日のページを綴る ==============
with tab1:
    with st.container(border=True):
        # --- メイン入力（テキスト・写真）---
        text_memo = st.text_area(
            "📝 今の気持ちや、ふと感じたこと...",
            st.session_state.get("ai_memo", ""),
            height=120,
            placeholder="今日の空の色、誰かの笑顔、心が動いた瞬間..."
        )

        image_files = st.file_uploader(
            "📸 今日の1枚（最大6枚）",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        # アイデアB: EXIFから撮影日時を自動入力
        if image_files:
            try:
                first_img = Image.open(image_files[0])
                exif_data = first_img._getexif()
                if exif_data:
                    tag_map = {v: k for k, v in ExifTags.TAGS.items()}
                    date_tag = tag_map.get("DateTimeOriginal")
                    if date_tag and date_tag in exif_data:
                        exif_dt = datetime.strptime(exif_data[date_tag], "%Y:%m:%d %H:%M:%S")
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
                            "この写真の情景を、思い出に残るような少しエモーショナルな文章（150文字程度）で描写してください。"
                        ]
                    )
                    st.session_state["ai_memo"] = response.text.strip()
                    st.rerun()
                except Exception as e:
                    st.error(f"AIテキスト生成に失敗しました: {e}")

        # --- オプション（アコーディオン）---
        with st.expander("📎 その他の記憶も添える（日付・場所・音声など・任意）"):
            col_a, col_b = st.columns(2)
            with col_a:
                default_date = st.session_state.get("exif_date", datetime.today())
                date = st.date_input("🗓️ 日付", default_date)
            with col_b:
                location = st.text_input(
                    "📍 場所",
                    placeholder="お気に入りのカフェ、いつもの散歩道..."
                )

            pdf_file = st.file_uploader("PDF📄（旅程表やパンフなど 最大6ページ）", type=["pdf"])
            col1, col2 = st.columns(2)
            with col1:
                audio_file = st.file_uploader("現地音声🎙️（最大80秒）", type=["mp3", "wav", "m4a"])
            with col2:
                video_file = st.file_uploader("動画🎥（最大128秒）", type=["mp4", "mov"])

        # --- 保存ボタン ---
        if st.button("✨ この記憶のページを綴じる", type="primary", use_container_width=True):
            if image_files and len(image_files) > 6:
                st.warning("⚠️ 写真は最大6枚までです！最初の6枚のみを保存します。")
                image_files = image_files[:6]

            if not any([text_memo, image_files, audio_file, video_file, pdf_file]):
                st.error("⚠️ テキストか写真など、なにか1つ入力してください。")
            else:
                with st.spinner("あなたの記憶を優しくページに綴じています..."):
                    # 埋め込み生成（アップロード前にファイルポインタをリセット）
                    for f in (image_files or []):
                        f.seek(0)
                    if audio_file: audio_file.seek(0)
                    if video_file: video_file.seek(0)
                    if pdf_file: pdf_file.seek(0)

                    embedding = create_embedding(text_memo, image_files, audio_file, video_file, pdf_file)

                    if embedding:
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
                            "location": location or "",
                            "text_memo": text_memo,
                            "image_paths": img_urls,
                            "audio_path": audio_url or "",
                            "video_path": video_url or "",
                            "pdf_path": pdf_url or "",
                            "timestamp": datetime.now().isoformat(),
                            "embedding": embedding,
                        }).execute()

                        st.success("✅ 今日の記憶を優しく本に挟みました。")

# ============== タブ2: 記憶の糸をたぐる ==============
with tab2:
    st.subheader("記憶の糸をたぐり寄せる")
    st.write("言葉や1枚の写真から、過去のページをめくります。")

    query_text = st.text_input(
        "🔍 思い出したい空気感や、今の気分",
        placeholder="温かいコーヒー、雨の日の匂い、静かな夕暮れ..."
    )
    query_image = st.file_uploader("🖼️ 参考写真で検索（任意）", type=["jpg", "jpeg", "png"])
    threshold = st.slider(
        "💭 記憶のピント（左：ふんわり 〜 右：くっきり）",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05
    )

    if st.button("📖 記憶をめくる", type="primary", use_container_width=True):
        with st.spinner("あなたの記憶をめくっています..."):
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
                    st.info("指定した条件に合う記憶は見つかりませんでした。ピントを左（ふんわり）に動かしてみてください。")
                else:
                    st.write("### 📖 あなたの記憶のページ")
                    results.sort(key=lambda x: (x.get("date", ""), x.get("timestamp", "")))

                    for meta in results:
                        location_str = meta.get("location") or "ある場所で"
                        with st.container(border=True):
                            st.markdown(f"### 🍂 🗓️ {meta.get('date', '')} 📍 {location_str}")

                            # 画像グリッド（str/list 混在対応）
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
                                st.markdown(f"[📄 添付ファイルを開く]({pdf_url})")

                            if meta.get("text_memo"):
                                st.info(f"📝 {meta['text_memo']}")
