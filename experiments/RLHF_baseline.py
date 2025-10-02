# !pip install \
#   torch==2.0.1 \
#   trl==0.7.9 \
#   transformers==4.31.0 \
#   accelerate==0.21.0 \
#   huggingface_hub==0.16.4 \
#   diffusers==0.20.2 \
#   peft==0.4.0 \
#   datasets==2.14.4

# #install dependencies
# !pip install bitsandbytes --quiet  # for faster loading
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "sshleifer/tiny-gpt2"  # super small, fast for demo
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#load HH-RLHF dataset
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:300]")
print(dataset)

#train a simple reward model distilbert-style to predict preference (chosen > rejected)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# here you just take a default reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
# Ensure reward model is on the same device as the main model
reward_model = reward_model.to(model.device)

# reward_model = load(our_pretrained_model_from_other_file)

def preprocess_reward(example):
    return {
        "text": [example["chosen"], example["rejected"]],
        "label": [1, 0]
    }

raw_dataset = dataset.select(range(300))  # select a few examples

reward_data = []
for ex in raw_dataset:
    #print(ex, ex.keys())

    reward_data.append({"text": ex["chosen"], "label": 1.0})
    reward_data.append({"text": ex["rejected"], "label": 0.0})

reward_dataset = Dataset.from_list(reward_data)
reward_dataset = reward_dataset.train_test_split(test_size=0.1)

def tokenize_reward(example):
    return reward_tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_reward = reward_dataset.map(tokenize_reward, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="no",
    num_train_epochs=1,
    output_dir="./reward_model_output",
    logging_dir="./logs",
)

trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=tokenized_reward["train"],
    eval_dataset=tokenized_reward["test"],
)

# 
trainer.train()

#PPO fine-tuning with TRL
from trl import PPOTrainer, PPOConfig
import torch

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    bf16=False,
    fp16=False
)

class DummyProcessor:
    def __call__(self, samples):
        return samples

ppo_trainer = PPOTrainer(
    ppo_config,
    model=model,
    ref_model=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
    reward_model=reward_model,
    value_model=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
    train_dataset=tokenized_reward["train"],
    processing_class=DummyProcessor())

# Create mini prompt set for demo
prompts = ["What is the capital of France?", "Explain photosynthesis."] * 5

tokenizer.pad_token = tokenizer.eos_token

for prompt in prompts:
    query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    response_tensor = model.generate(query_tensor, max_new_tokens=30)
    full_input = torch.cat([query_tensor, response_tensor], dim=1)

    # Step 1: Decode tensor input into strings
    decoded_input = tokenizer.batch_decode(full_input, skip_special_tokens=True)

    # Step 2: Tokenize the strings
    reward_inputs = tokenizer(decoded_input, return_tensors="pt", padding=True, truncation=True)

    # Step 3: Move to model's device
    reward_inputs = {k: v.to(model.device) for k, v in reward_inputs.items()}

    # Step 4: Inference
    with torch.no_grad():
      reward = reward_model(**reward_inputs).logits[0]

    reward_score = torch.tensor([reward.item()])

    print(reward_score)
    # Use the correct method for PPOTrainer - step method with proper data format
    ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward_score])

#save models
model.save_pretrained("./ppo_rlhf_baseline")
tokenizer.save_pretrained("./ppo_rlhf_baseline")
reward_model.save_pretrained("./reward_model")
