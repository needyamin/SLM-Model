# slm-laravel-sql

Minimal from-scratch decoder-only Transformer training repo aimed at Laravel + SQL domain knowledge.

Steps:
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
