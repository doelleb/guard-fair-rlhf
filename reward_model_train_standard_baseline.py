########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional, Union

    # import evaluate
    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    # from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        HfArgumentParser,
        Trainer,
        TrainingArguments,
    )
    from transformers.utils import PaddingStrategy




    # Define and parse arguments.
    @dataclass
    class ScriptArguments:
        """
        These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
        """
        local_rank: Optional[int] = field(
            default=-1, metadata={"help": "Used for multi-gpu"})

        deepspeed: Optional[str] = field(
            default=None,
            metadata={
                "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
            },
        )
        per_device_train_batch_size: Optional[int] = field(default=1)
        per_device_eval_batch_size: Optional[int] = field(default=1)
        # for 8 GPU, the global batch size is 512
        gradient_accumulation_steps: Optional[int] = field(default=64)
        learning_rate: Optional[float] = field(default=2e-6)
        weight_decay: Optional[float] = field(default=0.001)

        # TOOD: replace this default with the tiny-gpt2 you found so that this can run on your laptop 
        model_name: Optional[str] = field(
            default="erwanf/gpt2-mini",
            metadata={
                "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
            },
        )
        bf16: Optional[bool] = field(
            default=False,
            metadata={"help": "Use bfloat16 if supported"}
        )
        fp16: Optional[bool] = field(
            default=True,   # ← enable float16 by default
            metadata={"help": "Use float16 mixed precision"}
        )

        num_train_epochs: Optional[int] = field(
            default=1,
            metadata={"help": "The number of training epochs for the reward model."},
        )
        train_set_path: Optional[str] = field(
            default="hendrydong/preference_700K",
            metadata={"help": "The dir of the subset of the training data to use"},
        )
        eval_set_path: Optional[str] = field(
            default="hendrydong/preference_700K",
            metadata={"help": "The dir of the subset of the eval data to use"},
        )
        output_path: Optional[str] = field(
            default="./models/llama3_rm",
            metadata={"help": "The dir for output model"},
        )
        gradient_checkpointing: Optional[bool] = field(
            default=True,
            metadata={"help": "Enables gradient checkpointing."},
        )
        optim: Optional[str] = field(
            # default="adamw_hf",
            default="adamw_torch",
            # default="adamw_torch_fused",
            metadata={"help": "The optimizer to use."},
        )
        lr_scheduler_type: Optional[str] = field(
            default="cosine",
            metadata={"help": "The lr scheduler"},
        )
        max_length: Optional[int] = field(default=4096)

        save_every_steps: Optional[int] = field(
            default=999999,
            metadata={"help": "Save the model every x steps"},
        )
        eval_every_steps: Optional[int] = field(
            default=999999,
            metadata={"help": "Eval the model every x steps"},
        )


    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    tokenizer_name = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # LOAD THE MODEL ONCE (so we can read n_ctx, then reuse it)
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if script_args.bf16 else None,
        #attn_implementation="flash_attention_2",  # ← enable FlashAttention v2, not avialable on windows, without having to install it via github repo
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)


    tokenizer.model_max_length = model.config.n_positions
    model.config.use_cache = not script_args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))



    # Get the dataset
    train_path = script_args.train_set_path
    eval_path = script_args.eval_set_path
    output_name = script_args.output_path


    def build_dataset(tokenizer, train_path, eval_path):
        def tokenize(sample):
            # recursively unwrap lists/tuples until we hit a scalar
            def unwrap(x):
                while isinstance(x, (list, tuple)) and len(x) > 0:
                    x = x[0]
                return x

            pos = unwrap(sample["chosen"])
            neg = unwrap(sample["rejected"])

            # coerce to str in case it's not already
            pos = pos if isinstance(pos, str) else str(pos)
            neg = neg if isinstance(neg, str) else str(neg)

            tok_pos = tokenizer(pos, truncation=True, max_length=tokenizer.model_max_length)
            tok_neg = tokenizer(neg, truncation=True, max_length=tokenizer.model_max_length)

            sample["input_ids_j"]      = tok_pos["input_ids"] 
            sample["attention_mask_j"] = tok_pos["attention_mask"]
            sample["input_ids_k"]      = tok_neg["input_ids"]
            sample["attention_mask_k"] = tok_neg["attention_mask"]
            return sample

        ds = load_dataset(train_path, split="train[:1000]").shuffle(seed=42) #only 100 questions are being used for training; check if syntax is working fine
        ds = ds.map(tokenize, num_proc=8)

        train_dataset = ds
        eval_dataset  = ds.select(range(100))
        return train_dataset, eval_dataset



    train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
    print(f"Loaded {len(train_dataset)} train examples, {len(eval_dataset)} eval examples")
    print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

    # Define the trainer
    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        eval_strategy="steps",
        eval_steps=script_args.eval_every_steps,
        save_strategy="steps",
        save_steps=script_args.save_every_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=0.03,
        report_to='wandb'
    )



    num_proc = 24  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names


    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: AutoTokenizer
        padding: Union[bool, str, PaddingStrategy] = "max_length"
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            merged_features = []

            for feature in features:
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_j"],
                        "attention_mask": feature["attention_mask_j"],
                    }
                )
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_k"],
                        "attention_mask": feature["attention_mask_k"],
                    }
                )
            ctx = self.tokenizer.model_max_length
            batch = self.tokenizer.pad(
                merged_features,
                padding="longest",           # ← use dynamic (“longest”) padding
                return_tensors=self.return_tensors,
            )


            # just in case any sequence slipped past, truncate last dim
            if batch["input_ids"].size(1) > ctx:
                batch["input_ids"]      = batch["input_ids"][:, -ctx:]
                batch["attention_mask"] = batch["attention_mask"][:, -ctx:]
            
            batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "return_loss": True,
            }
            return batch


    # Define the trainer
    def compute_metrics(eval_pred):
        result = {}
        pos_predictions_scores = eval_pred.predictions[0]
        neg_predictions_scores = eval_pred.predictions[1]
        # We assume that the first sample is preferred by default in groundtruth
        result['accuracy'] = np.sum(
            pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
        return result


    class FairnessLoss(nn.Module): 
        pass
    # loss for reward model: 
    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            rewards = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )[0]
            bsz = rewards.size(0)
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1

            # j is preferred, k is not 
            rewards_j = rewards[jidx]
            rewards_k = rewards[kidx]

            # loss is the bradley terry loss that tries to push likelihood of preferred over not preferred as high as possible 

            # TODO: potential challenge 
            # advay's fairness loss depends on "logging" certain objects such as embedding clusters, ... + all the RND stuff such as target network/predictor network (advay fill these values in) 
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean() # + some Fairness loss that depends on advay's code 
            if return_outputs:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
            return loss


    # Train the model, woohoo.
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=script_args.max_length),
    )

    print("Starting reward-model training…")
    trainer.train()


    print("Saving last checkpoint of the model")
    trainer.save_model(output_name + "/last_checkpoint")
    tokenizer.save_pretrained(output_name + "/last_checkpoint")

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import gaussian_kde
    from datasets import load_dataset

    model.eval()
    sns.set_theme(style="whitegrid")

    # grab 100 HH-RLHF pairs
    pairs = load_dataset("Dahoas/full-hh-rlhf", split="train") \
            .shuffle(seed=42).select(range(100))

    def get_reward_score(text: str) -> float:
        # clamp to the model’s actual context size
        max_len = tokenizer.model_max_length
        toks = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            return model(**toks).logits.squeeze().cpu().item()

    chosen_scores   = np.array([get_reward_score(ex["chosen"])   for ex in pairs])
    rejected_scores = np.array([get_reward_score(ex["rejected"]) for ex in pairs])

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))

    # plot KDEs
    sns.kdeplot(
        chosen_scores,
        label="Helpful",
        fill=True,
        alpha=0.5,
        linewidth=2
    )
    sns.kdeplot(
        rejected_scores,
        label="Harmless",
        fill=True,
        alpha=0.5,
        linewidth=2
    )

    plt.xlabel("Rewards")
    plt.ylabel("Density")
    plt.title("Reward Distribution: Helpful vs. Harmless")
    plt.legend()
    plt.tight_layout()
    plt.show()
