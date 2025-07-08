"""
Debug script to diagnose gradient flow issues with LoRA + quantized models.
"""
import torch
from load_model import ModelLoader
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def debug_model_gradients():
    """Debug gradient setup step by step."""
    
    print("🔍 DEBUGGING GRADIENT FLOW")
    print("=" * 50)
    
    # Step 1: Load quantized model
    print("\n1️⃣ Loading 8-bit quantized model...")
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    loader = ModelLoader(model_name)
    model, tokenizer = loader.load_quantized_model(load_in_8bit=True)
    
    print(f"   Model type: {type(model)}")
    print(f"   Model training mode: {model.training}")
    
    # Check base model parameters
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
    
    print(f"   Base model total params: {total_params}")
    print(f"   Base model trainable params: {trainable_params}")
    
    # Step 2: Prepare for k-bit training
    print("\n2️⃣ Preparing model for k-bit training...")
    try:
        model = prepare_model_for_kbit_training(model)
        print("   ✅ prepare_model_for_kbit_training() successful")
    except Exception as e:
        print(f"   ❌ prepare_model_for_kbit_training() failed: {e}")
        return
    
    # Step 3: Apply LoRA
    print("\n3️⃣ Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    try:
        peft_model = get_peft_model(model, lora_config)
        print("   ✅ LoRA application successful")
    except Exception as e:
        print(f"   ❌ LoRA application failed: {e}")
        return
    
    # Step 4: Check LoRA parameters
    print("\n4️⃣ Checking LoRA parameters...")
    lora_params = 0
    total_trainable = 0
    
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            total_trainable += 1
            if 'lora_' in name:
                lora_params += 1
                print(f"   ✅ {name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")
    
    print(f"\n   Total trainable parameters: {total_trainable}")
    print(f"   LoRA parameters: {lora_params}")
    
    if lora_params == 0:
        print("   ❌ PROBLEM: No LoRA parameters found!")
        return
    
    # Step 5: Test forward pass and gradient computation
    print("\n5️⃣ Testing gradient computation...")
    peft_model.train()
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 10)).cuda()
    
    try:
        # Forward pass
        outputs = peft_model(dummy_input, labels=dummy_input)
        loss = outputs.loss
        print(f"   ✅ Forward pass successful, loss: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")
        
        # Backward pass
        loss.backward()
        print("   ✅ Backward pass successful!")
        
        # Check if gradients were computed
        grad_count = 0
        for name, param in peft_model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        print(f"   Parameters with gradients: {grad_count}")
        
        if grad_count == 0:
            print("   ❌ PROBLEM: No gradients computed!")
        else:
            print("   ✅ SUCCESS: Gradients computed successfully!")
            
    except Exception as e:
        print(f"   ❌ Gradient computation failed: {e}")
        print(f"   Error type: {type(e)}")
        
        # Additional debugging
        print("\n🔍 Additional debugging:")
        print(f"   Model device: {next(peft_model.parameters()).device}")
        print(f"   Input device: {dummy_input.device}")
        print(f"   Model dtype: {next(peft_model.parameters()).dtype}")

if __name__ == "__main__":
    debug_model_gradients() 