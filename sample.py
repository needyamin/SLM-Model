# sample.py
import yaml, torch
import sentencepiece as spm
from model import DecoderOnlyTransformer
from pathlib import Path

cfg = yaml.safe_load(open("config.yaml"))
sp = spm.SentencePieceProcessor(model_file=cfg["io"]["tokenizer_model"])
vocab_size = sp.get_piece_size()

# build model and load checkpoint
mcfg = cfg["model"]
model = DecoderOnlyTransformer(vocab_size=vocab_size,
                               n_layers=mcfg["n_layers"],
                               d_model=mcfg["d_model"],
                               n_heads=mcfg["n_heads"],
                               d_ff=mcfg["d_ff"],
                               seq_len=cfg["data"]["seq_len"])

# Check if checkpoint exists
ckpt = Path(cfg["io"]["out_dir"]) / "model_final.pt"
if not ckpt.exists():
    print(f"Checkpoint not found: {ckpt}")
    print("Please train the model first using train.py")
    exit(1)

try:
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    print("✓ Model checkpoint loaded successfully")
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    exit(1)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✓ Model moved to device: {device}")

def generate(prompt_text, max_new_tokens=200):
    try:
        ids = sp.encode(prompt_text, out_type=int)
        if len(ids) >= cfg["data"]["seq_len"] - max_new_tokens:
            print(f"Warning: prompt too long ({len(ids)} tokens), truncating...")
            ids = ids[:cfg["data"]["seq_len"] - max_new_tokens - 1]
        
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(input_ids)
                next_logits = logits[0, -1, :]
                # greedy
                next_id = int(torch.argmax(next_logits).item())
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                if next_id == sp.eos_id():
                    break
        
        return sp.decode(input_ids[0].tolist())
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: {e}"

# Example prompt
prompt = "<|system|>\nYou are a Laravel + SQL assistant.\n\n<|user|>\nInstruction: Create a migration for users table.\n\n<|assistant|>\n"
print("Generating response...")
response = generate(prompt, max_new_tokens=200)
print("\nGenerated response:")
print(response)
