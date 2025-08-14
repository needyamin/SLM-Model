# slm-laravel-sql

Minimal from-scratch decoder-only Transformer training repo aimed at Laravel + SQL domain knowledge.


# Laravel + SQL Small Language Model (SLM)

A custom small-scale language model built **from scratch** for answering questions about Laravel, PHP, and SQL (PostgreSQL, SQLite).  
This model is designed to run on a single GPU and can be trained with your own curated dataset.

---

## 📜 Project Overview
This is a **decoder-only Transformer** model implemented in PyTorch.  
It is **NOT** based on any pre-trained weights — training starts from random initialization.

The dataset is intended to be **legal and safe**, using only:
- Laravel docs (MIT License)
- PostgreSQL docs (permissive)
- SQLite docs (Public Domain)
- Your own example Q&A

For MySQL and other restrictive sources, we recommend using RAG instead of embedding them into the model.

---

## 🧠 Model Architecture

**Type:** GPT-style Decoder-only Transformer  
**Positional Encoding:** Rotary Position Embeddings (RoPE)  
**Normalization:** RMSNorm  
**Activation:** SwiGLU  
**Attention:** Multi-Head Self-Attention (causal mask)  
**Initialization:** Xavier uniform

| Parameter        | Value               |
|------------------|---------------------|
| Layers           | 4                   |
| Hidden Size      | 256                 |
| Feedforward Size | 1024                |
| Attention Heads  | 4                   |
| Vocabulary Size  | 5000 (custom BPE)   |
| Max Sequence Len | 512 tokens          |
| Total Params     | ~8 million          |

---

## ⚙️ Training Details

### Optimization
- **Optimizer:** AdamW  
- **Learning Rate:** 3e-4 (cosine decay, warmup 2%)  
- **Batch Size:** 32 sequences × 512 tokens  
- **Weight Decay:** 0.01  
- **Gradient Clipping:** 1.0  
- **Precision:** Mixed FP16/BF16

### Training Strategy
1. **Tokenizer Training** — train a custom BPE tokenizer on your Laravel + SQL dataset  
2. **Dataset Preprocessing** — chunk docs into ≤512 token segments  
3. **Causal Language Modeling (CLM)** — model predicts the next token  
4. **Validation Split** — 95% train / 5% val  
5. **Checkpoint Saving** — every N steps

---

## 🖥 System Requirements

| Component      | Minimum        | Recommended      |
|----------------|----------------|------------------|
| Python         | 3.10           | 3.11+            |
| GPU VRAM       | 4 GB           | 8+ GB            |
| RAM            | 8 GB           | 16+ GB           |
| Disk Space     | 2 GB           | 5+ GB            |
| OS             | Linux/Windows  | Linux (CUDA)     |

---

## 📦 Installation

```bash
git clone https://github.com/yourname/slm-laravel-sql.git
cd slm-laravel-sql
pip install -r requirements.txt


## 1) Train the Tokenizer
```python train_tokenizer.py --data data/ --vocab-size 5000```

## 2) Train the Model
```python train.py --config config.yaml```

## 3) Generate Text
```python generate.py --prompt "In Laravel, how do I validate a unique email?"```


## 📚 License

- Model code: MIT License
- Dataset: Follow respective licenses for each included source
- Laravel: MIT
- PostgreSQL: PostgreSQL License
- SQLite: Public Domain
- MySQL: Not embedded — use RAG

🔮 Next Steps
- Add LoRA adapters for low-VRAM fine-tuning
- Increase model depth to 8–12 layers for better accuracy
- Integrate RAG for MySQL + latest Laravel versions


## Steps:
1. Put your `.txt` data files into `data/raw/`. Each file can contain instruction/answer pairs and code blocks.
2. Train tokenizer:
   ```
   python tokenizer_train.py
   ```
3. Build dataset shards:
   ```
   python build_dataset.py
   ```
4. Train the model:
   ```
   python train.py
   ```
5. Generate examples:
   ```
   python sample.py
   ```

⚠️ Only use data you are licensed to use. This is a minimal starter; for production you'll want optimizations (FlashAttention, gradient checkpointing, distributed training, better sampling).
