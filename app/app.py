"""
Single-file Flask app to run your LSTM text generator.

How it works
- Loads the checkpoint at ./app/saved_model/checkpoint.pt (same path you used).
- Builds the LSTMModel, loads model + optimizer state (and moves optimizer tensors to device).
- Provides a web UI with:
    1) Top input box for seed text
    2) Temperature input (with label above telling allowed range 0.1 - 0.9)
       - If the value is out of range, a clear error message is shown and model is not run.
    3) Run button
    4) Results area that shows the generated text. If no temperature is supplied, the page shows example outputs for [0.5, 0.6, 0.7, 0.8].

Run:
    python flask_lstm_app.py

Requirements (example):
    pip install flask torch torchvision
    # If your tokenization requires nltk:
    pip install nltk
    python -m nltk.downloader punkt

Drop this file next to your project and start the server.
"""

import os
import re
import torch
import torch.nn as nn
from flask import Flask, request, render_template_string
from LSTM import LSTMModel

#  Utility: tokenization fallback 
try:
    from nltk.tokenize import word_tokenize
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False

    def word_tokenize(text):
        # Very small fallback tokeniser: splits on whitespace but keeps punctuation as tokens
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

#  Load checkpoint and build model once on startup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVED_PATH = os.path.join(os.getcwd(),"saved_model")
CKPT_PATH = os.path.join(SAVED_PATH, "checkpoint.pt")

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}. Put checkpoint.pt there.")

ckpt = torch.load(CKPT_PATH, map_location=device)

# sometimes idx2word keys are strings; coerce to ints for safety
idx2word_raw = ckpt.get("idx2word", {})
try:
    idx2word = {int(k): v for k, v in idx2word_raw.items()}
except Exception:
    idx2word = idx2word_raw

word2idx = ckpt.get("word2idx", {})

vocab_size = len(word2idx)

model = LSTMModel(
    vocab_size=vocab_size,
    embed_size=ckpt.get("embed_size", 256),
    hidden_size=ckpt.get("hidden_size", 256),
    num_layers=ckpt.get("num_layers", 2),
    dropout=ckpt.get("dropout", 0.3)
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-5
)

# Load model + optimizer state if present
if "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
elif "model" in ckpt:
    model.load_state_dict(ckpt["model"])  

if "optimizer_state" in ckpt:
    optimizer.load_state_dict(ckpt["optimizer_state"])

   
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

model.eval()

#  Generation function 
def generate_text(
    model, word2idx, idx2word, seed_text,
    length=50, temperature=0.8, sequence_length=50
):
    model.eval()

    tokens = word_tokenize(seed_text.lower())
    indices = [word2idx.get(token, word2idx.get("<unk>", 0)) for token in tokens]
    generated = indices.copy()

    with torch.no_grad():
        for _ in range(length):
            current_seq = generated[-sequence_length:]

            if len(current_seq) < sequence_length:
                current_seq = [word2idx.get("<pad>", 0)] * (sequence_length - len(current_seq)) + current_seq

            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)

            output = model(input_tensor)

            logits = output[0, -1] / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

    # Join by space; fallback for missing idx
    return " ".join(idx2word.get(idx, "<unk>") for idx in generated)

#  Flask app & template 
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LSTM Text Generator</title>
  <style>
    body{font-family:system-ui, -apple-system, Roboto, 'Segoe UI', Arial; margin:24px;}
    .container{max-width:900px;margin:0 auto}
    header{margin-bottom:18px}
    input[type=text], input[type=number], textarea{width:100%;padding:10px;margin:6px 0;border-radius:6px;border:1px solid #ccc}
    label{font-weight:600}
    .row{display:flex;gap:12px}
    .col{flex:1}
    .btn{background:#2563eb;color:white;border:none;padding:10px 14px;border-radius:8px;cursor:pointer}
    .error{color:#b91c1c;margin-top:8px}
    pre{background:#111827;color:#e5e7eb;padding:12px;border-radius:8px;white-space:pre-wrap}
    .sep{margin:10px 0;border-top:1px solid #e5e7eb}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>LSTM Text Generator</h1>
      <p>Enter a seed text and a temperature (0.1 to 0.9). If temperature is empty, example temps will be shown.</p>
    </header>

    <form method="post">
      <div>
        <label for="seed">Seed text</label>
        <input id="seed" name="seed" type="text" value="{{ seed|e }}" placeholder="Write some seed text... (e.g. 'he was')" required />
      </div>

      <div style="margin-top:8px">
        <label for="temperature">Temperature (tell: from 0.1 to 0.9)</label>
        <input id="temperature" name="temperature" type="text" value="{{ temperature|e }}" placeholder="0.5" />
        {% if temp_error %}
          <div class="error">{{ temp_error }}</div>
        {% endif %}
      </div>

      <div style="margin-top:12px">
        <button class="btn" type="submit">Generate</button>
      </div>
    </form>

    <div class="sep"></div>

    {% if results %}
      <h2>Results</h2>
      {% for r in results %}
        <div style="margin-bottom:12px">
          <strong>{{ r.temp }}</strong>
          <div style="margin-top:6px">--------------------</div>
          <pre>{{ r.text }}</pre>
        </div>
      {% endfor %}
    {% else %}
      <p>No results yet — enter seed text and press Generate.</p>
    {% endif %}

    <footer style="margin-top:28px;color:#6b7280">
      <small>Model loaded from: {{ ckpt_path }}</small>
    </footer>
  </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    seed = "he was"
    temperature = ""
    temp_error = None
    results = []

    if request.method == 'POST':
        seed = request.form.get('seed', '').strip() or seed
        temperature = request.form.get('temperature', '').strip()

        # If temperature is provided, validate it
        if temperature:
            # allow comma or dot decimal separators
            temp_str = temperature.replace(',', '.')
            try:
                t = float(temp_str)
                if not (0.1 <= t <= 0.9):
                    temp_error = 'Temperature must be between 0.1 and 0.9 (inclusive).'
                else:
                    # valid single temperature -> generate one result
                    gen = generate_text(model, word2idx, idx2word, seed, length=50, temperature=t, sequence_length=50)
                    results.append({"temp": f"{t}", "text": f'Generated sequence from "{seed}":\n{gen}'})
            except ValueError:
                temp_error = 'Temperature must be a number (e.g. 0.5)'
        else:
            # No temperature provided -> show sample outputs for 0.5,0.6,0.7,0.8
            sample_temps = [0.5, 0.6, 0.7, 0.8]
            for t in sample_temps:
                gen = generate_text(model, word2idx, idx2word, seed, length=50, temperature=t, sequence_length=50)
                results.append({"temp": f"{t}", "text": f'Generated sequence from "{seed}":\n{gen}'})

    return render_template_string(
        TEMPLATE,
        seed=seed,
        temperature=temperature,
        temp_error=temp_error,
        results=results,
        ckpt_path=CKPT_PATH
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
