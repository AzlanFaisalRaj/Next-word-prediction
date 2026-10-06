# Next Word Prediction with LSTM

A next-word predictor trained on famous quotes, served through a Streamlit app. Type the start of a sentence and the model returns the most likely next word, the top alternatives with their probabilities, and an optional multi-word continuation.

<!-- Add a screenshot or GIF here: ![demo](assets/demo.png) -->
<img width="950" height="434" alt="image" src="https://github.com/user-attachments/assets/6773e96a-c422-446a-b986-9aa66451662b" />


## What it does

- Predicts the next word for any sentence prefix.
- Shows the top-K candidate words with probabilities, so you can see how confident the model is.
- Continues a sentence for up to 25 words using greedy decoding.
- Runs locally with a single command.

## How it works

1. **Data:** about 3,000 quotes from `qoute_dataset.csv`, lowercased with punctuation removed.
2. **Tokenization:** Keras `Tokenizer` with a 10,000-word vocabulary.
3. **Training pairs:** every quote is split into prefixes. Each prefix is the input, and the word that follows it is the label. Inputs are pre-padded to the longest prefix (745 tokens).
4. **Model:** `Embedding (50)` → `LSTM (128)` → `Dense (10,000, softmax)`, trained with categorical cross-entropy and Adam.
5. **Inference:** the app applies the same preprocessing as training, pads to the saved `max_len`, and reads the softmax output.

The notebook also defines a `SimpleRNN` version of the same architecture for comparison. Only the LSTM was saved and is used by the app.

## Project structure

```
├── app.py                  # Streamlit UI and inference code
├── lstm_model.h5           # Trained LSTM model
├── tokenizer.pkl           # Fitted Keras tokenizer
├── max_len.pkl             # Input length used during training
├── qoute_dataset.csv       # Quotes dataset
├── codefile.ipynb          # Preprocessing, training, saving
├── RNNimplementation.ipynb # SimpleRNN walkthrough on a small sentiment example
└── .streamlit/config.toml  # Dark theme settings
```

## Run it locally

```bash
git clone (https://github.com/AzlanFaisalRaj/Next-word-prediction.git)
cd D:\DL Projects\Next word prediction
pip install streamlit tensorflow numpy
streamlit run app.py
```

The `.h5` file was saved with Keras 3, so use TensorFlow 2.16 or newer. Keep `app.py`, the model, `tokenizer.pkl`, `max_len.pkl`, and the `.streamlit` folder in the same directory.

## Limitations

- **Small data:** about 3,000 quotes is not much. The model learns common quote phrasing, not general English.
- **Greedy generation:** continuation always picks the single most likely word, so longer outputs often repeat or drift.
- **Fixed vocabulary:** words outside the 10,000-word vocabulary are ignored.
- **Full-width padding:** every input is padded to 745 tokens, which is wasteful for short sentences. Inference is still fast on CPU for a single query, but it is not optimized.
- **No evaluation numbers here:** I haven't added a held-out test set or perplexity score yet.

## Ideas for next steps

- Temperature or top-k sampling for more natural generated text.
- A validation split with perplexity and top-k accuracy.
- Masking instead of padding to 745, or a shorter max length.
- A Transformer baseline for comparison.

## Tech stack

Python, TensorFlow/Keras, NumPy, Streamlit.
