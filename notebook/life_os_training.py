# ⚠️ Run this script in Google Colab with T4 GPU enabled.

import os
import torch
import psutil
import builtins
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 注入 psutil 防止 Unsloth 在 Colab 报错
builtins.psutil = psutil

# 2. 安装/加载 Unsloth (如果是在本地运行需确保已安装)
from unsloth import FastLanguageModel, is_bfloat16_supported

def main():
    # === 配置 ===
    max_seq_length = 2048
    dtype = None # Auto detection
    load_in_4bit = True 
    data_file = "data/generated_data_10k.json"

    # === 加载模型 ===
    print("Loading Llama-3 Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # === 添加 LoRA 适配器 ===
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # === 准备数据 (Alpaca 格式) ===
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token 
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    print(f"Loading dataset from {data_file}...")
    dataset = load_dataset("json", data_files=data_file, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # === 训练参数 ===
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Demo purpose, increase for real training
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    )
    
    # 手动设置进程数，绕过 Bug
    training_args.dataset_num_proc = 2

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = training_args,
    )

    # === 开始训练 ===
    print("🚀 Starting Training...")
    trainer.train()
    print("✅ Training Completed!")

    # === 保存模型 ===
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

if __name__ == "__main__":
    main()