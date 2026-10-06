import html
import os
import pickle
import string

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Next Word Prediction", page_icon="✍️", layout="centered")

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600;700&display=swap');
:root{--bg:#16132B;--card:#221E3F;--line:#3A3566;--text:#EDE9FF;--muted:#9C96C9;--amber:#FFC145;--teal:#5EEAD4;}
html,body,.stApp{background:var(--bg);font-family:'Manrope',sans-serif;color:var(--text);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0;}
.block-container{max-width:760px;padding-top:3rem;padding-bottom:3rem;}

.hero h1{font-family:'Instrument Serif',serif;font-weight:400;font-size:3.4rem;line-height:1.05;margin:0 0 .6rem;color:var(--text);letter-spacing:-.01em;}
.hero p{color:var(--muted);font-size:1.02rem;margin:0 0 2rem;max-width:34rem;line-height:1.55;}

/* input */
[data-testid="stTextInput"] input{background:var(--card);border:1px solid var(--line);border-radius:14px;color:var(--text);
  font-family:'Manrope',sans-serif;font-size:1.05rem;padding:.9rem 1.1rem;}
[data-testid="stTextInput"] input:focus{border-color:var(--amber);box-shadow:0 0 0 3px rgba(255,193,69,.18);}
[data-testid="stTextInput"] label{display:none;}

/* buttons */
.stButton>button{border-radius:999px;font-family:'Manrope',sans-serif;font-weight:600;transition:background .15s,border-color .15s;}
button[data-testid="stBaseButton-secondary"],button[kind="secondary"]{background:transparent;border:1px solid var(--line);color:var(--muted);font-size:.85rem;padding:.2rem .9rem;}
button[data-testid="stBaseButton-secondary"]:hover,button[kind="secondary"]:hover{border-color:var(--amber);color:var(--amber);background:transparent;}
button[data-testid="stBaseButton-primary"],button[kind="primary"]{background:var(--amber);border:none;color:#2A1F00;padding:.55rem 1.4rem;}
button[data-testid="stBaseButton-primary"]:hover,button[kind="primary"]:hover{background:#FFD16E;color:#2A1F00;}

/* quote card */
.quote-card{position:relative;background:var(--card);border:1px solid var(--line);border-radius:22px;padding:2.2rem 2rem 2rem 2.4rem;margin:1.6rem 0 1.2rem;}
.qmark{position:absolute;left:1.1rem;top:-.2rem;font-family:'Instrument Serif',serif;font-size:5rem;color:var(--amber);opacity:.5;line-height:1;}
.quote-text{font-family:'Instrument Serif',serif;font-size:2.1rem;line-height:1.3;margin:0;word-wrap:break-word;}
.hint{color:var(--muted);font-size:.9rem;margin:.9rem 0 0;}
.pred{color:#2A1F00;background:var(--amber);border-radius:8px;padding:0 .4em;font-style:italic;animation:sweep .7s ease-out both;}
@keyframes sweep{from{background-size:0 100%;opacity:.2}to{opacity:1}}
.caret{display:inline-block;width:2px;height:1.6rem;background:var(--teal);margin-left:3px;vertical-align:middle;animation:blink 1.1s steps(1) infinite;}
@keyframes blink{50%{opacity:0}}
.gen{color:var(--amber);}

/* probability bars */
.section-title{font-size:1.05rem;font-weight:700;margin:1.6rem 0 .8rem;}
.bar-row{display:grid;grid-template-columns:7.5rem 1fr 3.6rem;gap:.8rem;align-items:center;margin-bottom:.55rem;}
.bar-word{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.bar-track{background:rgba(255,255,255,.06);border-radius:999px;height:10px;overflow:hidden;}
.bar-fill{height:100%;border-radius:999px;background:var(--teal);opacity:.75;}
.bar-row.top .bar-fill{background:var(--amber);opacity:1;}
.bar-pct{color:var(--muted);font-size:.88rem;text-align:right;font-variant-numeric:tabular-nums;}

.foot{color:var(--muted);font-size:.8rem;margin-top:3rem;border-top:1px solid var(--line);padding-top:1rem;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:transparent;}
@media (max-width:600px){.hero h1{font-size:2.5rem}.quote-text{font-size:1.6rem}.bar-row{grid-template-columns:5rem 1fr 3.2rem}}
@media (prefers-reduced-motion:reduce){.pred,.caret{animation:none}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
MODEL_CANDIDATES = ["lstm_model.h5", "lstm_model (1).h5", "lstm_model__1_.h5"]


@st.cache_resource
def load_resources():
    path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Model file not found. Looked for: {', '.join(MODEL_CANDIDATES)}")
    model = load_model(path)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_to_word = {i: w for w, i in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_word


try:
    model, tokenizer, max_len, index_to_word = load_resources()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

PUNCT = str.maketrans("", "", string.punctuation)


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
def top_predictions(text, k=5):
    """Return [(word, probability)] for the k most likely next words."""
    text = text.lower().translate(PUNCT)
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return []
    # Model was trained with input length == max_len
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    probs = model.predict(seq, verbose=0)[0]
    out = []
    for idx in np.argsort(probs)[::-1][: k + 5]:
        word = index_to_word.get(int(idx))
        if word:
            out.append((word, float(probs[idx])))
        if len(out) == k:
            break
    return out


def generate(text, n_words):
    words = []
    current = text
    for _ in range(n_words):
        preds = top_predictions(current, 1)
        if not preds:
            break
        words.append(preds[0][0])
        current += " " + preds[0][0]
    return words


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Finish the sentence.</h1>
  <p>Type the start of a quote and an LSTM, trained on famous quotes, guesses the word that comes next.</p>
</div>
""",
    unsafe_allow_html=True,
)

EXAMPLES = ["the only way to", "life is what", "to be or not", "i think therefore"]


def set_example(text):
    st.session_state.user_text = text


st.text_input("Your text", key="user_text", placeholder="Type the start of a sentence, then press Enter…")

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=set_example, args=(ex,), use_container_width=True)

with st.expander("Settings"):
    top_k = st.slider("Alternative words to show", 3, 8, 5)
    n_words = st.slider("Words to generate", 3, 25, 10)

text = st.session_state.get("user_text", "").strip()

if not text:
    st.markdown(
        '<div class="quote-card"><span class="qmark">“</span>'
        '<p class="quote-text" style="color:var(--muted)">Your sentence will appear here<span class="caret"></span></p></div>',
        unsafe_allow_html=True,
    )
else:
    preds = top_predictions(text, top_k)
    safe = html.escape(text)
    if not preds:
        st.markdown(
            f'<div class="quote-card"><span class="qmark">“</span><p class="quote-text">{safe}</p>'
            '<p class="hint">None of these words are in the model\'s vocabulary. Try a more common phrase.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        best_word, _ = preds[0]
        st.markdown(
            f'<div class="quote-card"><span class="qmark">“</span>'
            f'<p class="quote-text">{safe} <span class="pred">{html.escape(best_word)}</span></p></div>',
            unsafe_allow_html=True,
        )

        top_p = preds[0][1] or 1.0
        rows = "".join(
            f'<div class="bar-row{" top" if i == 0 else ""}"><span class="bar-word">{html.escape(w)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{max(p / top_p * 100, 2):.1f}%"></div></div>'
            f'<span class="bar-pct">{p * 100:.1f}%</span></div>'
            for i, (w, p) in enumerate(preds)
        )
        st.markdown(f'<div class="section-title">Other likely words</div>{rows}', unsafe_allow_html=True)

        st.write("")
        if st.button("Continue the quote", type="primary"):
            with st.spinner("Writing…"):
                new_words = generate(text, n_words)
            st.markdown(
                f'<div class="quote-card"><span class="qmark">“</span><p class="quote-text">{safe} '
                f'<span class="gen">{html.escape(" ".join(new_words))}</span></p>'
                '<p class="hint">Greedy decoding: each step picks the single most likely word, so repeats are normal for a small model.</p></div>',
                unsafe_allow_html=True,
            )

st.markdown(
    f'<div class="foot">Embedding → LSTM → Softmax over {model.output_shape[-1]:,} words · context window of {max_len} words</div>',
    unsafe_allow_html=True,
)