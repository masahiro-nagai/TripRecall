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
    @import url('https://fonts.googleapis.com/css2?family=Shippori+Mincho:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

    span.material-symbols-rounded {
        display: none !important;
    }

    /* Base Font: Zen Maru Gothic for body, Shippori Mincho for Headings */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Zen Maru Gothic', sans-serif !important;
        color: #E2E2E7 !important;
    }
    
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-1104qqp,  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Shippori Mincho', serif !important;
        color: #F2CA50 !important; 
        font-weight: 600 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background overrides */
    .stApp, .main {
        background-color: #111317 !important;
    }

    /* Glassmorphism containers with No-Line rule */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(30, 32, 36, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 24px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
        padding: 1.5rem !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5) !important;
    }

    /* Primary Buttons - Gradient Gold */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #f2ca50, #d4af37) !important;
        color: #3c2f00 !important;
        border-radius: 50px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5) !important;
        filter: brightness(1.1);
    }
    
    /* Secondary Buttons - Glassy Gold */
    button[data-testid="baseButton-secondary"] {
        background-color: rgba(212, 175, 55, 0.1) !important;
        color: #D4AF37 !important;
        border: 1px solid rgba(212, 175, 55, 0.2) !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background-color: rgba(212, 175, 55, 0.2) !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
    }

    /* File Dropzone - Subtle Dark Gold */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #4D4635 !important;
        border-radius: 16px !important;
        background-color: rgba(26, 28, 32, 0.6) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #D4AF37 !important;
        background-color: rgba(26, 28, 32, 0.8) !important;
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stDateInput input {
        background-color: #333539 !important;
        color: #E2E2E7 !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus, .stDateInput input:focus {
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        background-color: #37393D !important;
    }

    [data-testid="stExpanderToggleIcon"] {
        display: none !important;
    }
    
    /* Tab Styling Override for sophisticated look */
    [data-baseweb="tab-list"] {
        gap: 10px;
    }
    [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 10px 20px !important;
        border-bottom: 2px solid transparent !important;
    }
    [aria-selected="true"] {
        color: #f2ca50 !important;
        border-bottom-color: #d4af37 !important;
    }
    
    /* Information boxes (st.info, st.success, etc) */
    [data-testid="stAlert"] {
        background-color: rgba(26, 28, 32, 0.8) !important;
        border: 1px solid #4D4635 !important;
        color: #E2E2E7 !important;
        border-radius: 16px !important;
    }
    
    /* Checkbox text */
    [data-testid="stCheckbox"] label p {
        color: #E2E2E7 !important;
    }
</style>
""", unsafe_allow_html=True)

ui_texts = {
    "ja": {
        "title": "📖 MomentWeave",
        "subtitle": "毎日のささやかな瞬間、感情、空気感を優しく織りなす、あなただけの記憶のノートです。",
        "tab_compose": "🖋️ 今日のページを綴る",
        "tab_search": "🔍 記憶の糸をたぐる",
        "memo_label": "📝 今の気持ちや、ふと感じたこと...",
        "memo_placeholder": "今日の空の色、誰かの笑顔、心が動いた瞬間...",
        "photo_label": "📸 今日の1枚（最大6枚）",
        "ai_button": "✨ 写真からAIに思い出を綴ってもらう",
        "ai_loading": "AIが情景を描写中...",
        "ai_prompt": "この写真の情景を、思い出に残るような少しエモーショナルな文章（150文字程度）で描写してください。",
        "options_toggle": "📎 その他の記憶も添える（日付・場所・音声など・任意）",
        "date_label": "🗓️ 日付",
        "location_label": "📍 場所",
        "location_placeholder": "お気に入りのカフェ、いつもの散歩道...",
        "pdf_label": "PDF📄（旅程表やパンフなど 最大6ページ）",
        "audio_label": "現地音声🎙️（最大80秒）",
        "video_label": "動画🎥（最大128秒）",
        "save_button": "✨ この記憶のページを綴じる",
        "err_photo_max": "⚠️ 写真は最大6枚までです！最初の6枚のみを保存します。",
        "err_empty": "⚠️ テキストか写真など、なにか1つ入力してください。",
        "save_loading": "あなたの記憶を優しくページに綴じています...",
        "save_success": "✅ 今日の記憶を優しく本に挟みました。",
        "search_title": "記憶の糸をたぐり寄せる",
        "search_subtitle": "言葉や1枚の写真から、過去のページをめくります。",
        "search_label": "🔍 思い出したい空気感や、今の気分",
        "search_placeholder": "温かいコーヒー、雨の日の匂い、静かな夕暮れ...",
        "search_image": "🖼️ 参考写真で検索（任意）",
        "search_slider": "💭 記憶のピント（左：ふんわり 〜 右：くっきり）",
        "search_button": "📖 記憶をめくる",
        "search_loading": "あなたの記憶をめくっています...",
        "search_empty": "指定した条件に合う記憶は見つかりませんでした。ピントを左（ふんわり）に動かしてみてください。",
        "search_results_title": "### 📖 あなたの記憶のページ",
        "default_location": "ある場所で",
        "open_pdf": "📄 添付ファイルを開く"
    },
    "en": {
        "title": "📖 MomentWeave",
        "subtitle": "A gentle notebook for weaving together everyday fleeting moments, feelings, and atmospheres.",
        "tab_compose": "🖋️ Weave Today's Page",
        "tab_search": "🔍 Trace the Threads",
        "memo_label": "📝 Feelings and fleeting thoughts...",
        "memo_placeholder": "The color of the sky today, someone's smile, a moving moment...",
        "photo_label": "📸 Today's Photos (Max 6)",
        "ai_button": "✨ Let AI weave memories from the photo",
        "ai_loading": "AI is depicting the scene...",
        "ai_prompt": "Describe the scene in this photo using an emotional and memorable tone (around 50 words).",
        "options_toggle": "📎 Add other memories (Date, Location, Audio, etc. - Optional)",
        "date_label": "🗓️ Date",
        "location_label": "📍 Location",
        "location_placeholder": "Your favorite cafe, the usual walking path...",
        "pdf_label": "📄 PDF (Itineraries, brochures - Max 6 pages)",
        "audio_label": "🎙️ Audio (Max 80s)",
        "video_label": "🎥 Video (Max 128s)",
        "save_button": "✨ Bind this memory page",
        "err_photo_max": "⚠️ Maximum 6 photos allowed! The first 6 will be saved.",
        "err_empty": "⚠️ Please provide at least text or a photo.",
        "save_loading": "Gently binding your memory into the pages...",
        "save_success": "✅ Today's memory has been gently tucked into the book.",
        "search_title": "Trace the Threads of Memory",
        "search_subtitle": "Flip through past pages from a word or a single photo.",
        "search_label": "🔍 Scents, moods, or atmospheres to recall",
        "search_placeholder": "Warm coffee, the smell of rain, a quiet dusk...",
        "search_image": "🖼️ Search with a reference photo (Optional)",
        "search_slider": "💭 Memory Focus (Left: Soft/Vague ~ Right: Sharp)",
        "search_button": "📖 Flip through memories",
        "search_loading": "Flipping through your memories...",
        "search_empty": "No memories matched your search. Try moving the focus to the left (softer).",
        "search_results_title": "### 📖 Your Memory Pages",
        "default_location": "somewhere",
        "open_pdf": "📄 Open attached file"
    }
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "ja"

t = ui_texts[st.session_state["lang"]]

col1, col2 = st.columns([8, 2])
with col1:
    st.title(t["title"])
with col2:
    lang_sel = st.selectbox("Language / 言語", ["ja", "en"], index=0 if st.session_state["lang"] == "ja" else 1, label_visibility="collapsed")
    if lang_sel != st.session_state["lang"]:
        st.session_state["lang"] = lang_sel
        st.rerun()

st.markdown(t["subtitle"])

tab1, tab2 = st.tabs([t["tab_compose"], t["tab_search"]])

# ============== タブ1: 今日のページを綴る ==============
with tab1:
    with st.container(border=True):
        # --- メイン入力（テキスト・写真）---
        text_memo = st.text_area(
            t["memo_label"],
            st.session_state.get("ai_memo", ""),
            height=120,
            placeholder=t["memo_placeholder"]
        )

        image_files = st.file_uploader(
            t["photo_label"],
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
        if st.button(t["ai_button"], disabled=not image_files):
            with st.spinner(t["ai_loading"]):
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
                            t["ai_prompt"]
                        ]
                    )
                    st.session_state["ai_memo"] = response.text.strip()
                    st.rerun()
                except Exception as e:
                    st.error(f"AIテキスト生成に失敗しました: {e}")

        # --- オプション（チェックボックストグル）---
        date = datetime.today()
        location = ""
        audio_file = None
        video_file = None
        pdf_file = None
        show_optional = st.checkbox(t["options_toggle"])
        if show_optional:
            col_a, col_b = st.columns(2)
            with col_a:
                default_date = st.session_state.get("exif_date", datetime.today())
                date = st.date_input(t["date_label"], default_date)
            with col_b:
                location = st.text_input(
                    t["location_label"],
                    placeholder=t["location_placeholder"]
                )

            pdf_file = st.file_uploader(t["pdf_label"], type=["pdf"])
            col1_a, col2_a = st.columns(2)
            with col1_a:
                audio_file = st.file_uploader(t["audio_label"], type=["mp3", "wav", "m4a"])
            with col2_a:
                video_file = st.file_uploader(t["video_label"], type=["mp4", "mov"])

        # --- 保存ボタン ---
        if st.button(t["save_button"], type="primary", use_container_width=True):
            if image_files and len(image_files) > 6:
                st.warning(t["err_photo_max"])
                image_files = image_files[:6]

            if not any([text_memo, image_files, audio_file, video_file, pdf_file]):
                st.error(t["err_empty"])
            else:
                with st.spinner(t["save_loading"]):
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

                        st.success(t["save_success"])

# ============== タブ2: 記憶の糸をたぐる ==============
with tab2:
    st.subheader(t["search_title"])
    st.write(t["search_subtitle"])

    query_text = st.text_input(
        t["search_label"],
        placeholder=t["search_placeholder"]
    )
    query_image = st.file_uploader(t["search_image"], type=["jpg", "jpeg", "png"])
    threshold = st.slider(
        t["search_slider"],
        min_value=0.0, max_value=1.0, value=0.4, step=0.05
    )

    if st.button(t["search_button"], type="primary", use_container_width=True):
        with st.spinner(t["search_loading"]):
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
                    st.info(t["search_empty"])
                else:
                    st.write(t["search_results_title"])
                    results.sort(key=lambda x: (x.get("date", ""), x.get("timestamp", "")))

                    for meta in results:
                        location_str = meta.get("location") or t["default_location"]
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
                                st.markdown(f"[{t['open_pdf']}]({pdf_url})")

                            if meta.get("text_memo"):
                                st.info(f"📝 {meta['text_memo']}")
