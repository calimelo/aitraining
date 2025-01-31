import openai
import json
import time

# 🔑 Set Your OpenAI API Key
openai.api_key = "your_openai_api_key"

# 📂 Step 1: Prepare the Fine-Tuning Dataset
dataset = [
    {"messages": [
        {"role": "system", "content": "You are a helpful AI customer support agent."},
        {"role": "user", "content": "My internet is not working."},
        {"role": "assistant", "content": "I'm sorry to hear that! Have you tried restarting your router?"}
    ]},
    {"messages": [
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "You can reset your password by clicking 'Forgot Password' on the login page."}
    ]},
    {"messages": [
        {"role": "user", "content": "Can I cancel my subscription?"},
        {"role": "assistant", "content": "Yes, you can cancel anytime in your account settings."}
    ]}
]

# Save dataset as JSONL file
dataset_filename = "fine_tuning_dataset.jsonl"
with open(dataset_filename, "w") as f:
    for entry in dataset:
        json.dump(entry, f)
        f.write("\n")

print(f"✅ Dataset saved as {dataset_filename}")

# 📤 Step 2: Upload Dataset to OpenAI
upload_response = openai.File.create(
    file=open(dataset_filename, "rb"),
    purpose='fine-tune'
)
file_id = upload_response["id"]
print(f"✅ Dataset uploaded with File ID: {file_id}")

# 🚀 Step 3: Start Fine-Tuning
fine_tune_response = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)
fine_tune_job_id = fine_tune_response["id"]
print(f"🚀 Fine-tuning started! Job ID: {fine_tune_job_id}")

# ⏳ Step 4: Monitor Fine-Tuning Progress
print("📡 Monitoring fine-tuning process...")
while True:
    job_status = openai.FineTuningJob.retrieve(fine_tune_job_id)
    status = job_status["status"]
    print(f"🕒 Current Status: {status}")

    if status in ["completed", "failed", "cancelled"]:
        break
    time.sleep(30)  # Check status every 30 seconds

print(f"✅ Fine-tuning finished with status: {status}")

# 🎯 Step 5: Use the Fine-Tuned Model
fine_tuned_model = job_status.get("fine_tuned_model")
if fine_tuned_model:
    print(f"✅ Your fine-tuned model ID: {fine_tuned_model}")

    # Generate response from the fine-tuned model
    response = openai.ChatCompletion.create(
        model=fine_tuned_model,
        messages=[{"role": "user", "content": "How do I reset my password?"}]
    )

    print("💬 Fine-Tuned Model Response:", response["choices"][0]["message"]["content"])
else:
    print("❌ Fine-tuning failed. Check your OpenAI dashboard.")
