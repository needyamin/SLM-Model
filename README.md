# SLM-Model: Small Language Model for Code Generation

A lightweight, efficient transformer-based language model specifically designed for code generation tasks, particularly Laravel and SQL assistance. This model implements a decoder-only transformer architecture optimized for small-scale training and inference.

## 🏗️ **Model Architecture**

### **Core Components**

- **Decoder-Only Transformer**: Based on GPT-style architecture without encoder
- **Multi-Head Self-Attention**: Causal attention mechanism for autoregressive generation
- **Positional Embeddings**: Learnable positional encodings
- **Layer Normalization**: Applied before attention and feed-forward layers
- **GELU Activation**: Gaussian Error Linear Unit for feed-forward networks

### **Model Specifications**

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `n_layers` | 12 | Number of transformer blocks |
| `d_model` | 512 | Hidden dimension size |
| `n_heads` | 8 | Number of attention heads |
| `d_ff` | 2048 | Feed-forward hidden dimension |
| `seq_len` | 128 | Maximum sequence length |
| `vocab_size` | 500 | Vocabulary size (configurable) |

### **Technical Details**

- **Attention Mechanism**: Scaled dot-product attention with causal masking
- **Optimization**: AdamW optimizer with cosine learning rate scheduling
- **Mixed Precision**: Automatic Mixed Precision (AMP) support for faster training
- **Gradient Clipping**: Norm-based gradient clipping at 1.0
- **Memory Efficient**: Streaming data loading with sharded datasets

## 🚀 **Features**

- ✅ **Lightweight**: Small model size suitable for resource-constrained environments
- ✅ **Fast Training**: Optimized for quick iteration and experimentation
- ✅ **Code-Focused**: Pre-trained on Laravel and SQL examples
- ✅ **Easy Deployment**: Simple inference pipeline
- ✅ **Customizable**: Easy to modify architecture and training parameters
- ✅ **Cross-Platform**: Works on CPU and GPU (CUDA)

## 📋 **Requirements**

### **System Requirements**

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: Optional but recommended (CUDA 11.8+)
- **Storage**: 2GB free space

### **Dependencies**

All required packages are listed in `requirements.txt`:

```bash
torch>=2.0.0          # PyTorch deep learning framework
sentencepiece>=0.1.98  # Tokenization
numpy>=1.21.0         # Numerical computing
tqdm>=4.64.0          # Progress bars
pyyaml>=6.0           # Configuration parsing
transformers>=4.30.0  # Hugging Face transformers
datasets>=2.12.0      # Dataset handling
accelerate>=0.20.0    # Training acceleration
tensorboard>=2.12.0   # Training visualization
matplotlib>=3.5.0     # Plotting
seaborn>=0.11.0       # Statistical visualization
```

## 🛠️ **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/needyamin/SLM-Model
cd SLM-Model
```

### **2. Create Virtual Environment**

```bash
# Using venv (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n slm-model python=3.9
conda activate slm-model
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Verify Installation**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sentencepiece; print('SentencePiece installed successfully')"
```

## 📊 **Data Preparation**

### **1. Prepare Training Data**

Place your training text files in `data/raw/` directory. The model expects plain text files with code examples.

**Example data format:**
```
Instruction: Create a Laravel migration for users table.
Answer:
php artisan make:migration create_users_table
Schema::create('users', function (Blueprint $table) {
    $table->id();
    $table->string('name');
    $table->string('email')->unique();
    $table->timestamps();
});
```

### **2. Build Dataset**

```bash
python build_dataset.py
```

This will:
- Tokenize your raw text files
- Create sequence-packed numpy arrays
- Save processed shards in `data/processed/`

## ⚙️ **Configuration**

Edit `config.yaml` to customize your model and training:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  vocab_size: 500
  seq_len: 128

model:
  n_layers: 12
  d_model: 512
  n_heads: 8
  d_ff: 2048

training:
  micro_batch: 4
  grad_accum_steps: 8
  lr: 0.001
  weight_decay: 0.01
  warmup_steps: 500
  total_steps: 10000
  save_every: 1000
  device: "cuda"
  mixed_precision: true

io:
  out_dir: "out"
  checkpoint_prefix: "ckpt"
  tokenizer_model: "out/tokenizer.model"
```

## 🎯 **Training**

### **Start Training**

```bash
python train.py
```

### **Training Process**

1. **Data Loading**: Streams data from processed shards
2. **Forward Pass**: Computes logits and loss
3. **Backward Pass**: Computes gradients
4. **Optimization**: Updates model parameters
5. **Checkpointing**: Saves model state periodically

### **Training Features**

- **Gradient Accumulation**: Effective batch size = micro_batch × grad_accum_steps
- **Learning Rate Scheduling**: Warmup followed by cosine decay
- **Mixed Precision**: Automatic FP16 training for speed
- **Progress Monitoring**: Real-time loss tracking with tqdm

### **Monitoring Training**

```bash
# View training logs
tail -f out/training.log

# Use TensorBoard (if installed)
tensorboard --logdir out/
```

## 🔮 **Inference & Generation**

### **Generate Text**

```bash
python sample.py
```

### **Custom Generation**

```python
from sample import generate

# Generate code for a specific task
prompt = "Instruction: Create a Laravel model for User with relationships."
response = generate(prompt, max_new_tokens=200)
print(response)
```

### **Generation Parameters**

- **Temperature**: Control randomness (lower = more deterministic)
- **Max Tokens**: Maximum number of tokens to generate
- **Top-k Sampling**: Limit vocabulary choices
- **Beam Search**: Generate multiple candidates

## 📁 **Project Structure**

```
SLM-Model/
├── model.py              # Model architecture definition
├── train.py              # Training script
├── sample.py             # Inference and generation
├── build_dataset.py      # Data preprocessing
├── tokenizer_train.py    # Tokenizer training
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Data directory
│   ├── raw/            # Raw text files
│   └── processed/      # Processed numpy arrays
└── out/                # Output directory
    ├── checkpoints/    # Model checkpoints
    └── tokenizer.model # Trained tokenizer
```

## 🔧 **Customization**

### **Modify Model Architecture**

Edit `model.py` to change:
- Number of layers
- Hidden dimensions
- Attention heads
- Activation functions

### **Adjust Training Parameters**

Modify `config.yaml` for:
- Learning rate and schedule
- Batch sizes
- Training duration
- Regularization

### **Add New Features**

- **Custom Loss Functions**: Modify loss calculation in `train.py`
- **Data Augmentation**: Add preprocessing in `build_dataset.py`
- **Evaluation Metrics**: Implement custom evaluation functions

## 📈 **Performance & Optimization**

### **Training Speed**

- **CPU**: ~100-500 tokens/second
- **GPU (RTX 3080)**: ~1000-5000 tokens/second
- **GPU (V100)**: ~2000-10000 tokens/second

### **Memory Usage**

- **Model Parameters**: ~23M parameters
- **Training Memory**: 2-8GB depending on batch size
- **Inference Memory**: 1-2GB

### **Optimization Tips**

1. **Reduce Sequence Length**: Lower `seq_len` for faster training
2. **Use Mixed Precision**: Enable AMP for 2x speed improvement
3. **Gradient Accumulation**: Increase effective batch size without memory
4. **Data Sharding**: Process data in smaller chunks

## 🐛 **Troubleshooting**

### **Common Issues**

**CUDA Out of Memory**
```bash
# Reduce batch size in config.yaml
micro_batch: 2
grad_accum_steps: 16
```

**Training Loss Not Decreasing**
- Check learning rate (too high/low)
- Verify data quality
- Increase model capacity

**Slow Training**
- Enable mixed precision
- Use GPU if available
- Reduce sequence length

### **Debug Mode**

```bash
# Run with verbose logging
python train.py --debug

# Check model parameters
python -c "from model import DecoderOnlyTransformer; m = DecoderOnlyTransformer(500, 128); print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')"
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **PyTorch Team**: For the excellent deep learning framework
- **SentencePiece**: For efficient tokenization
- **Transformer Architecture**: Based on "Attention Is All You Need"
- **GPT Models**: Inspiration for decoder-only architecture

## 📞 **Support**

- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: Contact maintainers directly

---

**Happy Coding! 🚀**

*This model is designed to be a starting point for code generation tasks. Feel free to adapt and extend it for your specific use cases.*
