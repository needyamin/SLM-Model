# tokenizer_train.py
import sentencepiece as spm
import glob, os
from pathlib import Path
import yaml

cfg = yaml.safe_load(open("config.yaml"))
raw_dir = cfg["data"]["raw_dir"]
vocab_size = cfg["data"]["vocab_size"]
out = Path("out")
out.mkdir(exist_ok=True)

# Concatenate raw files into a single corpus file for SP training
corpus = out / "corpus.txt"
with corpus.open("w", encoding="utf-8") as w:
    for p in glob.glob(os.path.join(raw_dir, "*.txt")):
        with open(p, "r", encoding="utf-8", errors="ignore") as r:
            text = r.read()
            if text.strip():
                w.write(text.replace("\r\n", "\n") + "\n")

sp_model_prefix = str(out / "tokenizer")
spm.SentencePieceTrainer.Train(
    input=str(corpus),
    model_prefix=sp_model_prefix,
    vocab_size=vocab_size,
    model_type="bpe",
    input_sentence_size=1000000,
    shuffle_input_sentence=True,
    bos_id=-1, eos_id=2, pad_id=0, unk_id=1,
    user_defined_symbols=["<|system|>","<|user|>","<|assistant|>"]
)

print("Tokenizer trained ->", sp_model_prefix + ".model")
