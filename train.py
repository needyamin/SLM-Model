# train.py
import os, glob, time
import math, yaml
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from model import DecoderOnlyTransformer
import sentencepiece as spm

cfg = yaml.safe_load(open("config.yaml"))
device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
out_dir = cfg["io"]["out_dir"]
os.makedirs(out_dir, exist_ok=True)

# Load tokenizer
sp = spm.SentencePieceProcessor(model_file=cfg["io"]["tokenizer_model"])
vocab_size = sp.get_piece_size()

# Build model
mcfg = cfg["model"]
model = DecoderOnlyTransformer(vocab_size=vocab_size,
                               n_layers=mcfg["n_layers"],
                               d_model=mcfg["d_model"],
                               n_heads=mcfg["n_heads"],
                               d_ff=mcfg["d_ff"],
                               seq_len=cfg["data"]["seq_len"])
model = model.to(device)

# Optimizer & scheduler
training = cfg["training"]
optim = AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("mixed_precision", True))

# Load dataset shards
shards = sorted(glob.glob(os.path.join(cfg["data"]["processed_dir"], "shard_*.npy")))
if not shards:
    raise RuntimeError("No processed shards: run build_dataset.py first")
print("Found shards:", len(shards))

# helper to sample shards in streaming fashion
def shard_generator():
    while True:
        for sh in shards:
            arr = np.load(sh)
            # arr shape: (Nseq, seq_len)
            for i in range(len(arr)):
                yield torch.from_numpy(arr[i]).long()

g = shard_generator()
micro_batch = training["micro_batch"]
grad_accum = training["grad_accum_steps"]
total_steps = training["total_steps"]
save_every = training["save_every"]
warmup = training["warmup_steps"]

step = 0
epoch = 0
running_loss = 0.0
pbar = tqdm(total=total_steps)
model.train()

while step < total_steps:
    epoch += 1
    # each outer loop collects micro_batch * grad_accum sequences -> effective batch
    for _ in range( (micro_batch * grad_accum) ):
        try:
            seq = next(g).to(device)  # seq_len tensor
            # create input and target
            x = seq.unsqueeze(0)  # (1, T)
            y = x.clone()
            
            # forward
            with torch.cuda.amp.autocast(enabled=cfg["training"].get("mixed_precision", True)):
                logits = model(x)  # 1,T,V
                # Ensure logits and targets have correct shapes for cross_entropy
                logits_flat = logits.view(-1, logits.size(-1))  # (T, V)
                targets_flat = y.view(-1)  # (T,)
                loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                loss = loss / grad_accum
            
            scaler.scale(loss).backward()
            running_loss += loss.item()

        except Exception as e:
            print(f"Error during training step: {e}")
            continue

        # optimization step
        if (_ + 1) % grad_accum == 0:
            # lr schedule (simple warmup -> cosine)
            step += 1
            # compute lr multiplier
            if step < warmup:
                lr_mult = step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            for gparam in optim.param_groups:
                gparam["lr"] = training["lr"] * lr_mult

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            pbar.update(1)

            if step % 10 == 0:
                pbar.set_description(f"loss={running_loss / 10:.4f}")
                running_loss = 0.0

            if step % save_every == 0 or step == total_steps:
                ckpt_path = os.path.join(out_dir, f"{cfg['io']['checkpoint_prefix']}_step{step}.pt")
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "sp_model": cfg["io"]["tokenizer_model"]
                }, ckpt_path)
                print("Saved checkpoint:", ckpt_path)

            if step >= total_steps:
                break
    if step >= total_steps:
        break

pbar.close()
print("Training finished. Final step:", step)
torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
