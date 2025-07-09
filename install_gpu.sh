#!/bin/bash

# GPU Installation Script for Domain Name Generator
# This script installs PyTorch with CUDA support and bitsandbytes with GPU support

echo "🚀 Installing GPU-enabled dependencies for Domain Name Generator"
echo "================================================================"

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "⚠️  Warning: No conda environment detected. Make sure you're in the correct environment."
    echo "   Run: conda activate your_environment_name"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "❌ NVIDIA GPU not detected. This script is for GPU installations."
    exit 1
fi

# Step 1: Install PyTorch with CUDA support
echo ""
echo "📦 Step 1: Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install bitsandbytes with GPU support
echo ""
echo "📦 Step 2: Installing bitsandbytes with GPU support..."
pip install bitsandbytes>=0.40.0

# Step 3: Install remaining requirements
echo ""
echo "📦 Step 3: Installing remaining dependencies..."
pip install -r requirements.txt

# Step 4: Verify installations
echo ""
echo "🔍 Step 4: Verifying GPU support..."

# Test PyTorch CUDA
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA not available in PyTorch')
"

# Test bitsandbytes GPU support
python -c "
try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__}')
    
    # Test if GPU quantization is available
    if hasattr(bnb, 'Linear8bitLt'):
        print('✅ bitsandbytes GPU support: Available')
    else:
        print('❌ bitsandbytes GPU support: Not available')
        
except ImportError as e:
    print(f'❌ bitsandbytes import error: {e}')
except Exception as e:
    print(f'❌ bitsandbytes test error: {e}')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Run: python Model/download_models.py"
echo "   2. Test a model: python Model/model_v4.py"
echo ""
echo "🔧 If you encounter issues:"
echo "   - Make sure you're in the correct conda environment"
echo "   - Check that CUDA drivers are properly installed"
echo "   - Try: pip install --force-reinstall bitsandbytes" 