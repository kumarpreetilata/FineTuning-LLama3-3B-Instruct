***Fine-Tuning Meta LLaMA 3.2-3B Instruct with LoRA (Guanaco)***

**Overview**

This notebook provides a step-by-step guide for fine-tuning the Meta LLaMA 3.2-3B Instruct model using LoRA (Low-Rank Adaptation). It includes loading the base model, applying LoRA adapters, and merging them for inference.

**Requirements**

Before running the notebook, ensure you have the following dependencies installed:

pip install torch transformers peft accelerate bitsandbytes

**File Structure**

![image](https://github.com/user-attachments/assets/668f87c3-aaf6-4d94-9e27-d6493fa17544)


**Important Files**

adapter_model.safetensors: The fine-tuned LoRA weights.

adapter_config.json: Configuration for the LoRA adapter.

**Steps**

- Load the Base ModelThe script loads the pre-trained Meta LLaMA 3.2-3B Instruct model in FP16.

- Load LoRA WeightsThe fine-tuned LoRA adapter weights are loaded from adapter_model.safetensors.

- Merge LoRA with Base ModelThe LoRA parameters are merged into the base model for optimized inference.

- Save the Final ModelThe merged model is saved to disk for further usage.

**Code Example**

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Define model paths
model_name = "meta-llama/Llama-3.2-3B-Instruct"
lora_weights_path = "meta-llama/Llama-3.2-3B-Instruct/guanaco"

# Load base model in FP16
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, lora_weights_path)
model = model.merge_and_unload()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)



**Common Issues & Fixes**

1. OSError: No file named pytorch_model.bin or model.safetensors

Ensure you have downloaded the full model from Hugging Face.

If only LoRA weights (adapter_model.safetensors) exist, use PeftModel to load them.

2. CUDA Memory Issues

- Reduce batch size.

- Use torch_dtype=torch.float16 to save memory.

- Use device_map="auto" for efficient GPU memory allocation.

**References**

Hugging Face Transformers

PEFT: Parameter-Efficient Fine-Tuning



