#install dependencies
#!pip install trl datasets accelerate peft transformers --quiet
#!pip install bitsandbytes --quiet  # for faster loading

#load base model
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "sshleifer/tiny-gpt2"  # super small, fast for demo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#load HH-RLHF dataset
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/hh-rlhf", split="train[:300]")
print(dataset[0])

#train a simple reward model distilbert-style to predict preference (chosen > rejected)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# here you just take a default reward model 
reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

# reward_model = load(our_pretrained_model_from_other_file) 

def preprocess_reward(example):
    return {
        "text": [example["chosen"], example["rejected"]],
        "label": [1, 0]
    }

raw_dataset = dataset.select(range(300))  # select a few examples

reward_data = []
for ex in raw_dataset:
    reward_data.append({"text": ex["chosen"], "label": 1})
    reward_data.append({"text": ex["rejected"], "label": 0})

reward_dataset = Dataset.from_list(reward_data)
reward_dataset = reward_dataset.train_test_split(test_size=0.1)

def tokenize_reward(example):
    return reward_tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_reward = reward_dataset.map(tokenize_reward, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
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
trainer.train()

#PPO fine-tuning with TRL
from trl import PPOTrainer, PPOConfig
import torch

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=False,
)

ppo_trainer = PPOTrainer(ppo_config, model, tokenizer, reward_model=reward_model)

# Create mini prompt set for demo
prompts = ["What is the capital of France?", "Explain photosynthesis."] * 5

for prompt in prompts:
    query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    response_tensor = model.generate(query_tensor, max_new_tokens=30)
    full_input = torch.cat([query_tensor, response_tensor], dim=1)

    with torch.no_grad():
        reward = reward_model(tokenizer.batch_decode(full_input, skip_special_tokens=True)[0], return_dict=True).logits[0]

    reward_score = torch.tensor([reward.item()])
    ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward_score)

#save models
model.save_pretrained("./ppo_rlhf_baseline")
tokenizer.save_pretrained("./ppo_rlhf_baseline")
reward_model.save_pretrained("./reward_model")
