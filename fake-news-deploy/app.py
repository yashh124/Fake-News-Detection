import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="TruthLens — AI Detection",
    page_icon="🔬",
    layout="wide"
)

# -----------------------------
# Custom CSS — Light & Relaxed
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:       #f7f5f2;
    --bg2:      #ffffff;
    --bg3:      #f0ede8;
    --bg4:      #e8e4de;
    --indigo:   #4f46e5;
    --indigo-l: #eef2ff;
    --indigo-b: #c7d2fe;
    --teal:     #0d9488;
    --teal-l:   #f0fdfa;
    --teal-b:   #99f6e4;
    --rose:     #e11d48;
    --rose-l:   #fff1f2;
    --amber:    #d97706;
    --amber-l:  #fffbeb;
    --text:     #1c1917;
    --text2:    #44403c;
    --sub:      #78716c;
    --border:   #e7e2db;
    --shadow:   0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:0 4px 16px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.04);
    --r:        14px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="block-container"] {
    padding: 2rem 2.5rem !important;
    max-width: 1100px !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── typography ── */
h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif !important; color: var(--text) !important; }
p, li, label, span, div { font-family: 'DM Sans', sans-serif !important; }

/* ── tabs ── */
[data-testid="stTabs"] { margin-top: 1.5rem; }

[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}

[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: var(--sub) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 9px 22px !important;
    border: none !important;
    border-radius: 9px !important;
    transition: all 0.18s !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--bg2) !important;
    color: var(--indigo) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    color: var(--text2) !important;
    background: rgba(255,255,255,0.5) !important;
}

[data-testid="stTabsContent"] {
    background: transparent !important;
    border: none !important;
    padding: 24px 0 0 !important;
}

/* ── inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--bg2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.83rem !important;
    padding: 13px 16px !important;
    transition: all 0.2s !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
    outline: none !important;
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: #a8a29e !important; }

[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: var(--text2) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    margin-bottom: 4px !important;
}

/* ── ALL buttons ── */
[data-testid="stButton"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    border-radius: 10px !important;
    padding: 9px 20px !important;
    transition: all 0.18s !important;
    background: var(--bg2) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text2) !important;
    box-shadow: var(--shadow) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

[data-testid="stButton"] button:hover {
    background: var(--bg3) !important;
    border-color: #c5c0b8 !important;
    color: var(--text) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stButton"] button:active {
    transform: translateY(0) !important;
    box-shadow: var(--shadow) !important;
}

/* ── primary buttons ── */
[data-testid="stButton"] button[kind="primary"] {
    background: var(--indigo) !important;
    border-color: var(--indigo) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background: #4338ca !important;
    border-color: #4338ca !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--r) !important;
    transition: all 0.2s !important;
    padding: 4px !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--indigo-b) !important;
    background: var(--indigo-l) !important;
}

[data-testid="stFileUploader"] * {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    color: var(--sub) !important;
}

/* ── alerts ── */
[data-testid="stSuccess"] {
    background: #f0fdf4 !important;
    border: 1.5px solid #bbf7d0 !important;
    border-radius: var(--r) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #15803d !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stError"] {
    background: var(--rose-l) !important;
    border: 1.5px solid #fecdd3 !important;
    border-radius: var(--r) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--rose) !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stInfo"] {
    background: var(--indigo-l) !important;
    border: 1.5px solid var(--indigo-b) !important;
    border-radius: var(--r) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.84rem !important;
    color: var(--indigo) !important;
    padding: 14px 20px !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stWarning"] {
    background: var(--amber-l) !important;
    border: 1.5px solid #fde68a !important;
    border-radius: var(--r) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: var(--amber) !important;
    padding: 14px 20px !important;
    box-shadow: var(--shadow) !important;
}

/* ── image ── */
[data-testid="stImage"] img {
    border-radius: var(--r) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--indigo) !important; }

/* ── hr ── */
hr {
    border: none !important;
    border-top: 1.5px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #c5c0b8; }

/* ── column gaps ── */
[data-testid="stHorizontalBlock"] { gap: 14px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Example Data
# -----------------------------
example_fake_title   = "Government secretly installs mind control chips in citizens"
example_fake_article = """A viral claim circulating on social media suggests that the government has
installed microchips in citizens through vaccines in order to control thoughts.
Experts and officials have strongly denied the claim and stated there is no
scientific evidence supporting this theory."""

example_real_title   = "Scientists discover new method to improve solar panel efficiency"
example_real_article = """Researchers from a leading university have developed a new material that
significantly increases solar panel efficiency. The technology could make
renewable energy more affordable and accelerate the transition to clean power."""

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_text_model():
    model      = joblib.load("models/text/svm_model.pkl")
    vectorizer = joblib.load("models/text/tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_image_model():
    return tf.keras.models.load_model("models/image/cnn_model.keras")

text_model, vectorizer = load_text_model()
image_model            = load_image_model()
IMG_SIZE               = (128, 128)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("""
<div style='padding:4px 2px 20px'>
  <div style='font-family:DM Sans,sans-serif;font-size:1.3rem;font-weight:700;color:#1c1917;letter-spacing:-0.5px'>
    🔬 TruthLens
  </div>
  <div style='font-family:DM Mono,monospace;font-size:0.63rem;color:#a8a29e;letter-spacing:1px;margin-top:2px;text-transform:uppercase'>
    AI Misinformation Detector
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='background:#f7f5f2;border:1px solid #e7e2db;border-radius:12px;padding:14px 16px;margin-bottom:10px'>
  <div style='font-family:DM Mono,monospace;font-size:0.6rem;color:#a8a29e;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px'>Text Pipeline</div>
  <div style='display:flex;flex-wrap:wrap;gap:6px'>
    <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;font-family:DM Mono,monospace;font-size:0.7rem;color:#4f46e5;padding:4px 10px;font-weight:500'>TF-IDF</span>
    <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;font-family:DM Mono,monospace;font-size:0.7rem;color:#4f46e5;padding:4px 10px;font-weight:500'>Linear SVM</span>
  </div>
</div>

<div style='background:#f7f5f2;border:1px solid #e7e2db;border-radius:12px;padding:14px 16px;margin-bottom:10px'>
  <div style='font-family:DM Mono,monospace;font-size:0.6rem;color:#a8a29e;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px'>Image Pipeline</div>
  <div style='display:flex;flex-wrap:wrap;gap:6px'>
    <span style='background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;font-family:DM Mono,monospace;font-size:0.7rem;color:#0d9488;padding:4px 10px;font-weight:500'>Custom CNN</span>
    <span style='background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;font-family:DM Mono,monospace;font-size:0.7rem;color:#0d9488;padding:4px 10px;font-weight:500'>Batch Norm</span>
    <span style='background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;font-family:DM Mono,monospace;font-size:0.7rem;color:#0d9488;padding:4px 10px;font-weight:500'>Augmentation</span>
  </div>
</div>

<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px'>
  <div style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:12px;padding:12px;text-align:center'>
    <div style='font-family:DM Sans,sans-serif;font-size:1.4rem;font-weight:700;color:#4f46e5;line-height:1'>44K</div>
    <div style='font-family:DM Mono,monospace;font-size:0.57rem;color:#818cf8;letter-spacing:1px;text-transform:uppercase;margin-top:4px'>Articles</div>
  </div>
  <div style='background:#f0fdfa;border:1px solid #99f6e4;border-radius:12px;padding:12px;text-align:center'>
    <div style='font-family:DM Sans,sans-serif;font-size:1.4rem;font-weight:700;color:#0d9488;line-height:1'>2K</div>
    <div style='font-family:DM Mono,monospace;font-size:0.57rem;color:#2dd4bf;letter-spacing:1px;text-transform:uppercase;margin-top:4px'>Images</div>
  </div>
</div>

<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:12px 16px'>
  <div style='font-family:DM Mono,monospace;font-size:0.6rem;color:#a8a29e;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px'>SVM Accuracy</div>
  <div style='font-family:DM Sans,sans-serif;font-size:1.5rem;font-weight:700;color:#d97706;line-height:1'>99.55%</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='margin-top:20px;padding-top:14px;border-top:1px solid #e7e2db;font-family:DM Mono,monospace;font-size:0.6rem;color:#c5c0b8;text-align:center'>
Final Year Research Project — 2026
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div style='margin-bottom:1.5rem'>
  <div style='display:flex;align-items:center;gap:14px;margin-bottom:14px'>
    <div style='width:48px;height:48px;background:linear-gradient(135deg,#4f46e5,#0d9488);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;box-shadow:0 4px 14px rgba(79,70,229,0.2)'>🔬</div>
    <div>
      <div style='font-family:DM Sans,sans-serif;font-size:1.9rem;font-weight:700;letter-spacing:-1px;line-height:1;color:#1c1917'>TruthLens</div>
      <div style='font-family:DM Mono,monospace;font-size:0.63rem;color:#a8a29e;letter-spacing:2px;text-transform:uppercase;margin-top:3px'>Hybrid AI Detection System</div>
    </div>
  </div>
  <div style='display:flex;gap:8px;flex-wrap:wrap'>
    <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;font-family:DM Sans,sans-serif;font-size:0.78rem;color:#4f46e5;padding:4px 14px;font-weight:500'>SVM text classifier</span>
    <span style='background:#f0fdfa;border:1px solid #99f6e4;border-radius:999px;font-family:DM Sans,sans-serif;font-size:0.78rem;color:#0d9488;padding:4px 14px;font-weight:500'>CNN image detector</span>
    <span style='background:#faf5ff;border:1px solid #e9d5ff;border-radius:999px;font-family:DM Sans,sans-serif;font-size:0.78rem;color:#7c3aed;padding:4px 14px;font-weight:500'>No login required</span>
    <span style='background:#fff7ed;border:1px solid #fed7aa;border-radius:999px;font-family:DM Sans,sans-serif;font-size:0.78rem;color:#c2410c;padding:4px 14px;font-weight:500'>Real-time prediction</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📰  Text — Fake News Detection", "🖼  Image — Deepfake Detection"])

# =====================================================
# TAB 1 — TEXT DETECTION
# =====================================================
with tab1:

    st.markdown("""
    <div style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:14px;padding:18px 22px;margin-bottom:20px'>
      <div style='font-family:DM Sans,sans-serif;font-weight:600;font-size:1rem;color:#3730a3;margin-bottom:5px'>Analyze News Article</div>
      <div style='font-family:DM Sans,sans-serif;font-size:0.84rem;color:#6366f1;line-height:1.6'>
        Enter a news title and article body. The TF-IDF + SVM pipeline classifies content as
        <strong>REAL</strong> or <strong>FAKE</strong> with a confidence score.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Example loader buttons — each has a unique key
    colA, colB = st.columns(2)
    with colA:
        if st.button("⚠️  Load Fake News Example", key="btn_fake_text", use_container_width=True):
            st.session_state["t_title"]   = example_fake_title
            st.session_state["t_article"] = example_fake_article
            st.rerun()
    with colB:
        if st.button("✅  Load Real News Example", key="btn_real_text", use_container_width=True):
            st.session_state["t_title"]   = example_real_title
            st.session_state["t_article"] = example_real_article
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Input fields bound to session state via key
    title = st.text_input(
        "News Title",
        key="t_title",
        placeholder="e.g. Scientists discover cure for common cold…"
    )
    article = st.text_area(
        "Article Body",
        key="t_article",
        height=190,
        placeholder="Paste the full article content here…"
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Analyze News", key="btn_analyze_text", type="primary"):
        t_val = st.session_state.get("t_title", "").strip()
        a_val = st.session_state.get("t_article", "").strip()
        if not t_val or not a_val:
            st.warning("⚠️  Please enter both a title and article body before analyzing.")
        else:
            with st.spinner("Running TF-IDF + SVM pipeline…"):
                text       = t_val + " " + a_val
                vector     = vectorizer.transform([text])
                prediction = text_model.predict(vector)[0]
                score      = text_model.decision_function(vector)[0]
                confidence = min(abs(score) * 100, 99.9)

            if prediction == 1:
                st.success("✅  REAL NEWS — Content appears legitimate")
            else:
                st.error("❌  FAKE NEWS — Misinformation detected")
            st.info(f"Confidence Score: {confidence:.2f}%")


# =====================================================
# TAB 2 — IMAGE DETECTION
# =====================================================
with tab2:

    st.markdown("""
    <div style='background:#f0fdfa;border:1px solid #99f6e4;border-radius:14px;padding:18px 22px;margin-bottom:20px'>
      <div style='font-family:DM Sans,sans-serif;font-weight:600;font-size:1rem;color:#0f766e;margin-bottom:5px'>Detect Deepfake Image</div>
      <div style='font-family:DM Sans,sans-serif;font-size:0.84rem;color:#0d9488;line-height:1.6'>
        Upload a face image (JPG / PNG). The CNN model analyzes spatial features and classifies it as
        <strong>REAL</strong> or <strong>DEEPFAKE</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Example image buttons — unique keys
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📷  Load Real Image Example", key="btn_real_img", use_container_width=True):
            try:
                st.session_state["example_image"] = Image.open("examples/real.jpg")
                # Clear any uploaded file flag
                if "uploaded_cleared" in st.session_state:
                    del st.session_state["uploaded_cleared"]
            except FileNotFoundError:
                st.warning("⚠️  File not found: examples/real.jpg")
    with col2:
        if st.button("📷  Load Fake Image Example", key="btn_fake_img", use_container_width=True):
            try:
                st.session_state["example_image"] = Image.open("examples/fake.jpg")
                if "uploaded_cleared" in st.session_state:
                    del st.session_state["uploaded_cleared"]
            except FileNotFoundError:
                st.warning("⚠️  File not found: examples/fake.jpg")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        key="img_uploader"
    )

    # Determine which image to show
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        # Discard example when user uploads
        if "example_image" in st.session_state:
            del st.session_state["example_image"]
    elif "example_image" in st.session_state:
        img = st.session_state["example_image"]
    else:
        img = None

    if img is not None:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 2])

        with colA:
            st.image(img, caption="Selected image", use_container_width=True)

        with colB:
            if st.button("🔍  Analyze Image", key="btn_analyze_img", type="primary"):
                with st.spinner("Running CNN inference…"):
                    img_resized = img.resize(IMG_SIZE)
                    img_array   = image.img_to_array(img_resized)
                    img_array   = img_array / 255.0
                    img_array   = np.expand_dims(img_array, axis=0)
                    pred        = image_model.predict(img_array)
                    probability = pred[0][0]

                if probability > 0.5:
                    st.success("✅  REAL IMAGE — No manipulation detected")
                    st.info(f"Confidence: {probability * 100:.2f}%")
                else:
                    st.error("❌  DEEPFAKE — AI-generated face detected")
                    st.info(f"Confidence: {(1 - probability) * 100:.2f}%")
    else:
        st.markdown("""
        <div style='background:#f7f5f2;border:1.5px dashed #e7e2db;border-radius:14px;padding:40px;text-align:center;margin-top:8px'>
          <div style='font-size:2rem;margin-bottom:10px'>🖼️</div>
          <div style='font-family:DM Sans,sans-serif;font-size:0.9rem;color:#a8a29e;font-weight:500'>
            Upload an image or load an example above to get started
          </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------
# Footer
# -----------------------------
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1.5px solid #e7e2db;padding-top:18px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>
  <div style='font-family:DM Sans,sans-serif;font-size:0.95rem;font-weight:700;color:#44403c;letter-spacing:-0.3px'>TruthLens</div>
  <div style='font-family:DM Mono,monospace;font-size:0.62rem;color:#c5c0b8;letter-spacing:0.5px'>
    Hybrid Fake News &amp; Deepfake Detection &nbsp;·&nbsp; Final Year Research Project – 2026
  </div>
</div>
""", unsafe_allow_html=True)