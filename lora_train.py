import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import getpass
a = getpass.getpass("Enter Token for HuggingFace: ")
login(token=a)

# Set OpenAI API Key (Only needed for inference, not fine-tuning)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Load GPT-3.5 Turbo Model from OpenAI (Using HF Model Equivalent)
model_name = "mistralai/Mistral-7B-v0.1"  # Alternative open-source LLM
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA fine-tuning configuration
lora_config = LoraConfig(
    r=8,  # Low-rank adaptation size (smaller = more efficient)
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Prevents overfitting
    target_modules=["q_proj", "v_proj"]  # Apply LoRA only to attention layers
)

# Convert model to LoRA-adapted model
lora_model = get_peft_model(model, lora_config)

# Print trainable parameters
lora_model.print_trainable_parameters()
from transformers import TrainingArguments, Trainer

# Sample dataset (Supervised fine-tuning)
train_data = [
    {"input": "Summarize: The stock market saw a sharp rise today due to increased investments.", 
     "output": "The stock market surged today amid strong investments."},

    {"input": "Summarize: Researchers discovered a new species of deep-sea fish.", 
     "output": "A new deep-sea fish species was discovered by researchers."}
]

# Convert dataset to tokenized format
def tokenize_data(example):
    input_text = f"User: {example['input']} \nAI: {example['output']}"
    return tokenizer(input_text, truncation=True, padding="max_length", return_tensors="pt")

# Prepare dataset
train_dataset = [tokenize_data(entry) for entry in train_data]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt3.5_lora_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset
)

# Start fine-tuning
trainer.train()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = lora_model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test the fine-tuned model
print(generate_response("Summarize: The government passed a new law regulating AI ethics."))

# Save the fine-tuned model
lora_model.save_pretrained("./gpt3.5_lora_finetuned")