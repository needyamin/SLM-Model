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
ckpt = Path(cfg["io"]["out_dir"]) / "model_final.pt"
state = torch.load(ckpt, map_location="cpu")
model.load_state_dict(state)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate(prompt_text, max_new_tokens=200):
    ids = sp.encode(prompt_text, out_type=int)
    assert len(ids) < cfg["data"]["seq_len"] - max_new_tokens, "prompt too long"
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_logits = logits[0, -1, :]
        # greedy
        next_id = int(torch.argmax(next_logits).item())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == sp.eos_id():
            break
    return sp.decode(input_ids[0].tolist())

# Example prompt
prompt = "<|system|>\nYou are a Laravel + SQL assistant.\n\n<|user|>\nInstruction: Create a migration for users table.\n\n<|assistant|>\n"
print(generate(prompt, max_new_tokens=200))
