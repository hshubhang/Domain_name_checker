#!/usr/bin/env python3
"""
Standalone Model Download Script for Domain Name Generator

This script downloads all required models from Hugging Face to make the project
fully reproducible. Run this once after cloning the repository.

Usage:
    python Model/download_models.py

Dependencies:
    pip install huggingface_hub
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not found. Install with: pip install huggingface_hub")
    sys.exit(1)


class ModelDownloader:
    """Downloads all required models for the Domain Name Generator project."""
    
    def __init__(self):
        """Initialize the model downloader."""
        self.project_root = Path(__file__).parent.parent
        self.models_config = {
            "lora_adapters": {
                "repo_id": "hshubhang/domain-generator-lora",
                "versions": ["v1", "v2", "v4"],
                "local_path": self.project_root / "lora_adapters",
                "description": "LoRA adapters for domain generation"
            },
            "classifier": {
                "repo_id": "hshubhang/domain-gibberish-classifier", 
                "local_path": self.project_root / "Model" / "classifier_model_v3",
                "description": "Gibberish classifier model"
            }
        }
        
    def check_existing_models(self) -> Dict[str, bool]:
        """Check which models are already downloaded."""
        status = {}
        
        # Check LoRA adapters
        for version in self.models_config["lora_adapters"]["versions"]:
            version_path = self.models_config["lora_adapters"]["local_path"] / version
            adapter_file = version_path / "adapter_model.safetensors"
            config_file = version_path / "adapter_config.json"
            tokenizer_file = version_path / "tokenizer.json"
            
            status[f"lora_{version}"] = (
                adapter_file.exists() and 
                config_file.exists() and 
                tokenizer_file.exists()
            )
        
        # Check classifier
        classifier_path = self.models_config["classifier"]["local_path"]
        model_file = classifier_path / "model.safetensors"
        config_file = classifier_path / "config.json"
        tokenizer_file = classifier_path / "tokenizer.json"
        
        status["classifier"] = (
            model_file.exists() and 
            config_file.exists() and 
            tokenizer_file.exists()
        )
        
        return status
    
    def print_status(self):
        """Print current model download status."""
        print("🔍 Checking existing models...")
        status = self.check_existing_models()
        
        print(f"\n📊 Model Status:")
        for model, exists in status.items():
            icon = "✅" if exists else "❌"
            print(f"   {icon} {model}")
        
        return status
    
    def download_lora_adapters(self, force: bool = False):
        """Download LoRA adapters for all versions."""
        config = self.models_config["lora_adapters"]
        
        print(f"\n📥 Downloading {config['description']}...")
        print(f"   Repository: {config['repo_id']}")
        
        for version in config["versions"]:
            version_path = config["local_path"] / version
            
            # Check if already exists
            if not force and version_path.exists():
                adapter_file = version_path / "adapter_model.safetensors"
                if adapter_file.exists():
                    print(f"   ✅ {version} already exists, skipping...")
                    continue
            
            print(f"   📥 Downloading LoRA {version}...")
            
            try:
                # Create version directory
                version_path.mkdir(parents=True, exist_ok=True)
                
                # Download essential files for this version
                essential_files = [
                    "adapter_model.safetensors",
                    "adapter_config.json", 
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json"
                ]
                
                for file_name in essential_files:
                    try:
                        hf_hub_download(
                            repo_id=config["repo_id"],
                            filename=f"{version}/{file_name}",
                            local_dir=config["local_path"],
                            local_dir_use_symlinks=False
                        )
                        print(f"      ✅ {file_name}")
                    except Exception as e:
                        print(f"      ⚠️  {file_name}: {e}")
                
                print(f"   ✅ LoRA {version} downloaded successfully!")
                
            except Exception as e:
                print(f"   ❌ Failed to download LoRA {version}: {e}")
                continue
    
    def download_classifier(self, force: bool = False):
        """Download the gibberish classifier model."""
        config = self.models_config["classifier"]
        
        print(f"\n📥 Downloading {config['description']}...")
        print(f"   Repository: {config['repo_id']}")
        
        # Check if already exists
        if not force and config["local_path"].exists():
            model_file = config["local_path"] / "model.safetensors"
            if model_file.exists():
                print(f"   ✅ Classifier already exists, skipping...")
                return
        
        try:
            # Create directory
            config["local_path"].mkdir(parents=True, exist_ok=True)
            
            # Download the entire classifier repository
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=config["local_path"],
                local_dir_use_symlinks=False
            )
            
            print(f"   ✅ Classifier downloaded successfully!")
            
        except Exception as e:
            print(f"   ❌ Failed to download classifier: {e}")
    
    def download_all(self, force: bool = False):
        """Download all required models."""
        print("🚀 Domain Name Generator - Model Download")
        print("=" * 50)
        
        # Show current status
        initial_status = self.print_status()
        
        # Check if everything is already downloaded
        if not force and all(initial_status.values()):
            print("\n✅ All models already downloaded!")
            print("\nTo re-download everything, run:")
            print("   python Model/download_models.py --force")
            return
        
        # Download missing models
        print(f"\n🔄 Downloading missing models...")
        
        # Download LoRA adapters
        if not all([initial_status.get(f"lora_{v}", False) for v in ["v1", "v2", "v4"]]):
            self.download_lora_adapters(force=force)
        
        # Download classifier
        if not initial_status.get("classifier", False):
            self.download_classifier(force=force)
        
        # Final status check
        print(f"\n🏁 Download Complete!")
        final_status = self.print_status()
        
        if all(final_status.values()):
            print(f"\n✅ All models downloaded successfully!")
            print(f"📁 Models location:")
            print(f"   LoRA adapters: {self.models_config['lora_adapters']['local_path']}")
            print(f"   Classifier: {self.models_config['classifier']['local_path']}")
            print(f"\n🚀 You can now run the models!")
        else:
            print(f"\n⚠️  Some models failed to download. Check the error messages above.")
    
    def get_download_info(self):
        """Print information about what will be downloaded."""
        print("📋 Download Information")
        print("=" * 30)
        
        total_size = 0
        
        print(f"\n📦 LoRA Adapters (hshubhang/domain-generator-lora)")
        for version in ["v1", "v2", "v4"]:
            size = "52MB" if version == "v1" else "104MB"
            print(f"   • {version}: ~{size} (adapter + tokenizer)")
            total_size += 52 if version == "v1" else 104
        
        print(f"\n🤖 Gibberish Classifier (hshubhang/domain-gibberish-classifier)")
        print(f"   • DistilBERT model: ~255MB")
        total_size += 255
        
        print(f"\n📊 Total download size: ~{total_size}MB")
        print(f"💾 Local storage needed: ~{total_size + 50}MB")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download all required models for Domain Name Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python Model/download_models.py              # Download missing models
    python Model/download_models.py --force      # Re-download all models
    python Model/download_models.py --info       # Show download information
        """
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download all models (even if they exist)"
    )
    
    parser.add_argument(
        "--info",
        action="store_true", 
        help="Show download information without downloading"
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = ModelDownloader()
    
    if args.info:
        downloader.get_download_info()
        return
    
    # Download models
    try:
        downloader.download_all(force=args.force)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 