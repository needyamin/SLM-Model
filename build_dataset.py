# build_dataset.py
import sentencepiece as spm
import numpy as np
import glob, os
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm

cfg = yaml.safe_load(open("config.yaml"))
sp_model = cfg["io"]["tokenizer_model"] if cfg["io"].get("tokenizer_model") else "out/tokenizer.model"
seq_len = cfg["data"]["seq_len"]
raw_dir = cfg["data"]["raw_dir"]
proc_dir = Path(cfg["data"]["processed_dir"])
proc_dir.mkdir(parents=True, exist_ok=True)

sp = spm.SentencePieceProcessor(model_file=sp_model)

# Read and tokenize all, then pack into contiguous sequences
token_stream = []
for p in glob.glob(os.path.join(raw_dir, "*.txt")):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        continue
    ids = sp.encode(text, out_type=int)
    token_stream.extend(ids + [sp.eos_id()])  # add eos between docs

# Convert to numpy and pack
arr = np.array(token_stream, dtype=np.int32)
n_full = len(arr) // seq_len
arr = arr[: n_full * seq_len]
if n_full == 0:
    print("No full sequences were produced. Consider lowering seq_len or adding more data.")
else:
    arr = arr.reshape((n_full, seq_len))

    # Save as binary .npy per shard
    shard_size = 1000  # adjust shard size for memory
    for i in range(0, n_full, shard_size):
        shard = arr[i:i+shard_size]
        outp = proc_dir / f"shard_{i//shard_size:04d}.npy"
        np.save(outp, shard)
        print("Saved", outp, shard.shape)
    print("Total sequences:", n_full)
