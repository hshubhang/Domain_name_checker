# Domain Name Generator Project

## Overview

This project implements an AI-powered domain name generation system using fine-tuned **Llama 3.1 8B** models with LoRA (Low-Rank Adaptation) adapters. The system can generate relevant domain names for business descriptions while filtering out inappropriate content and gibberish.

### **Multiple Model Versions**
- **v0**: meta-llama 3 8b instruct model using replicate api for testing
- **v1**: No filtering (baseline model)  
- **v2**: Gibberish filtering 
- **v3**: Gibberish filtering using DISTILBERT transformer model and LoRA finetuned and trained with augmented data. No NSFW Filtering.
- **v4**: Gibberish + NSFW filtering (most comprehensive) LoRA finetuned and trained with even more augmented data.

### **Content Filtering**
- **NSFW Detection**: 
Blocks inappropriate content using better-profanity + keyword filtering
Uses profanity-profanity library + custom keyword detection as fallback.
- **Gibberish Classification**: DISTILBERT classifier to detect meaningless input
- **Safety Measures**: Multi-layer filtering for production safety

### **API Interface**
- **REST API**: Flask-based API with confidence scoring
- **Response Formats**: Success, blocked (NSFW), filtered (gibberish), error states
- **Heuristic Scoring**: 5-factor confidence scoring (relevance, clarity, professionalism, memorability, TLD quality)

### **LLM as a judge Evaluation**
- **GPT-4o Integration**: Objective content analysis and domain quality assessment
- **Performance Metrics**: Legitimate accuracy, filtering accuracy, over-filtering detection

### Model Infrastructure
- **Base Model**: Meta Llama 3.1 8B Instruct
- **Fine-tuning**: LoRA adapters for domain-specific training
- **Quantization**: 8-bit quantization for memory efficiency
- **Classification**: BERT-based gibberish detector

### Evaluation Pipeline
- **Dataset**: 100 entries (50 legitimate, 31 gibberish, 20 NSFW)
- **Metrics**: Classification accuracy, domain quality scoring, legitimate blocking count
- **Ground Truth**: Dataset labels take precedence for performance assessment

### 🚀 **Setup (First Time)**
```bash
# 1. Clone the repository
git clone https://github.com/hshubhang/Domain_name_checker
cd Domain_name_checker

# 2. Create conda environment  
conda create -n domain python=3.9
conda activate domain

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (one-time setup)
python Model/download_models.py
```
# 5. Open ai api key needed for evaluator and Replicate api key needed for model_v0 run

### **Ready to Use**
```bash
# Start API server
python api_endpoint.py

# Or test directly
cd Model/
python model_v4.py
```

## Model Download

This project uses models hosted on **Hugging Face** for reproducibility:

- **LoRA Adapters**: [hshubhang/domain-generator-lora](https://huggingface.co/hshubhang/domain-generator-lora)
- **Gibberish Classifier**: [hshubhang/domain-gibberish-classifier](https://huggingface.co/hshubhang/domain-gibberish-classifier)

### Automatic Download
```bash
# Download all models (~515MB total)
python download_models.py

# Check what will be downloaded
python download_models.py --info

# Force re-download
python download_models.py --force
```

### Manual Download (Alternative)
```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download hshubhang/domain-generator-lora --local-dir lora_adapters  
huggingface-cli download hshubhang/domain-gibberish-classifier --local-dir Model/classifier_model_v3
```

### Model and evaluation Output files

These files can be found in the model outputs and evaluation outputs folder respectively.

### Evaluator

Evaluator located in the global folder



### Core Requirements
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Key Packages
- **torch>=2.0.0** - PyTorch for model inference
- **transformers>=4.30.0** - Hugging Face model loading
- **peft>=0.4.0** - LoRA adapter support
- **bitsandbytes>=0.39.0** - 8-bit quantization
- **huggingface_hub>=0.17.0** - Model download support
- **openai>=1.0.0** - GPT evaluation API
- **flask>=2.3.0** - REST API server
- **better-profanity>=0.7.0** - NSFW content filtering
- **scikit-learn>=1.0.0** - Classification utilities

### Environment Setup
```bash
# Create conda environment
conda create -n domain python=3.9
conda activate domain

# Install CUDA support (if using GPU)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

### Development/Testing Environment
| Component | Specification |
|-----------|---------------|
| **CPU** | 4+ vCPU (Intel Xeon @ 2.20 GHz recommended) |
| **Memory** | 15+ GB RAM |
| **GPU** | NVIDIA L4 (24 GB VRAM) or equivalent |
| **Storage** | 99+ GB (models require ~20GB) |
| **Driver** | NVIDIA Driver 550.54.15+ |


## Project Structure

```
Domain_name_checker/
├── download_models.py             # Model download script (run once)
├── api_endpoint.py               # REST API server
├── Evaluator_final.py            # Evaluation script
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── Model/                        # Model implementations
│   ├── model_v4.py              # Latest model (gibberish + NSFW filtering)
    |-- model_v3.py              # Model with DISTILBERT based gibberish filtering
│   ├── model_v2.py              # Gibberish filtering hardcoded 
│   ├── model_v1.py              # No filtering (baseline)
│   ├── classifier_model_v3/     # Downloaded gibberish classifier
│   └── train_lora_pretokenized.py # Training script (not needed for inference)
├── Evaluator/                    # Legacy evaluation system
├── data/                         # Training and test datasets
│   └── dataset_injection_v4.jsonl # Evaluation dataset (100 entries)
├── lora_adapters/                # Downloaded LoRA weights
│   ├── v1/                      # Basic model adapters
│   ├── v2/                      # Improved model adapters  
│   └── v4/                      # Latest model adapters
├── Model Outputs/                # Generated domain results
└── evaluator outputs/            # Evaluation reports
```

## Usage

### 🚀 **API Server**
```bash
# Start API server
python api_endpoint.py

# Test endpoints
curl http://localhost:5000/health
curl "http://localhost:5000/generate?business_description=organic coffee shop"
```

### 📊 **Model Evaluation**
```bash
# Run evaluation on all models
cd Evaluator/
python Evaluator_final.py
```

### 🤖 **Batch Processing**
```bash
# Generate domains for dataset
cd Model/
python model_v4.py
```

## API Endpoints

### `GET /health`
**Response**: `{"status": "healthy"}`

### `POST /generate`
**Request**: 
```json
{"business_description": "organic coffee shop in downtown area"}
```

**Success Response**:
```json
{
  "suggestions": [
    {"domain": "organicbrew.com", "confidence": 0.92},
    {"domain": "downtowncoffee.org", "confidence": 0.87}
  ],
  "status": "success"
}
```

**Blocked Response**:
```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

## Model Performance

### Key Metrics
- **Legitimate Accuracy**: How well models serve real businesses
- **Filtering Accuracy**: How well models block inappropriate content  
- **Domain Quality**: GPT-4o assessment of generated domains
- **Legitimate Blocked**: Count of over-filtered legitimate businesses

### Training Data
- **Fine-tuning**: Custom business description → domain name pairs
- **Classification**: Balanced legitimate/gibberish dataset
- **Validation**: Hold-out test set for unbiased evaluation


