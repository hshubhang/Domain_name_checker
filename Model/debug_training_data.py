"""
Debug the training data pipeline to find what breaks gradient flow.
"""
import torch
from datasets import load_from_disk
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from load_model import ModelLoader
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def debug_training_data():
    """Debug the actual training data pipeline."""
    
    print("🔍 DEBUGGING TRAINING DATA PIPELINE")
    print("=" * 50)
    
    # Step 1: Load model (we know this works)
    print("\n1️⃣ Loading model...")
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    loader = ModelLoader(model_name)
    model, tokenizer = loader.load_quantized_model(load_in_8bit=True)
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.train()
    print("   ✅ Model setup complete")
    
    # Step 2: Load tokenized dataset
    print("\n2️⃣ Loading tokenized dataset...")
    try:
        dataset = load_from_disk("../data/tokenized_dataset")
        print(f"   ✅ Dataset loaded: {len(dataset)} examples")
        print(f"   Features: {dataset.features}")
        
        # Check a sample
        sample = dataset[0]
        print(f"   Sample keys: {sample.keys()}")
        for key, value in sample.items():
            if isinstance(value, list):
                print(f"   {key}: length={len(value)}, type={type(value[0]) if value else 'empty'}")
            else:
                print(f"   {key}: {type(value)}")
                
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        return
    
    # Step 3: Test data collator  
    print("\n3️⃣ Testing data collator...")
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad_token to: {tokenizer.pad_token}")
    
    print(f"   Tokenizer pad_token: {tokenizer.pad_token}")
    print(f"   Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Get a small batch
    batch_data = [dataset[i] for i in range(2)]  # Just 2 examples
    
    try:
        batch = data_collator(batch_data)
        print("   ✅ Data collator successful")
        print(f"   Batch keys: {batch.keys()}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            else:
                print(f"   {key}: {type(value)}")
                
    except Exception as e:
        print(f"   ❌ Data collator failed: {e}")
        return
    
    # Step 4: Test model with real batch
    print("\n4️⃣ Testing model with real training batch...")
    
    # Move batch to GPU
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    try:
        # Forward pass with real data
        outputs = peft_model(**batch)
        loss = outputs.loss
        print(f"   ✅ Forward pass successful, loss: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")
        
        if not loss.requires_grad:
            print("   ❌ PROBLEM: Loss doesn't require gradients!")
            return
            
        if loss.grad_fn is None:
            print("   ❌ PROBLEM: Loss has no gradient function!")
            return
        
        # Test backward pass
        loss.backward()
        print("   ✅ Backward pass successful!")
        
        # Check gradients
        grad_count = 0
        for name, param in peft_model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        print(f"   Parameters with gradients: {grad_count}")
        
        if grad_count == 0:
            print("   ❌ PROBLEM: No gradients computed with real data!")
        else:
            print("   ✅ SUCCESS: Real data training works!")
            
    except Exception as e:
        print(f"   ❌ Model with real data failed: {e}")
        print(f"   Error type: {type(e)}")
        
        # Debug the inputs
        print("\n🔍 Debugging inputs:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}, requires_grad={value.requires_grad}")
                if value.numel() < 20:  # Small tensors
                    print(f"   {key} values: {value}")
    
    # Step 5: Test with different batch sizes
    print("\n5️⃣ Testing with single example...")
    
    try:
        single_batch = data_collator([dataset[0]])
        single_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in single_batch.items()}
        
        outputs = peft_model(**single_batch)
        loss = outputs.loss
        print(f"   ✅ Single example works, loss: {loss.item():.4f}")
        
        loss.backward()
        print("   ✅ Single example backward pass successful!")
        
    except Exception as e:
        print(f"   ❌ Single example failed: {e}")

if __name__ == "__main__":
    debug_training_data() 