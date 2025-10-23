#!/usr/bin/env python3

import subprocess
import sys
import os

def setup_environment():
    """Installs the necessary packages for the script."""
    print("--- [Step 1] Setting up environment. Installing packages... ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth[cu121-py310]", "huggingface_hub", "datasets", "trl", "torch", "transformers", "accelerate"])
        print("--- Environment setup complete. ---")
    except subprocess.CalledProcessError as e:
        print(f"Error during package installation: {e}")
        sys.exit(1)

# === 1. CONFIGURATION ===
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
DO_SPACES_BUCKET = "my-model-assets"
TRAIN_FILE = "my_copywriting_data.jsonl"
EVAL_FILE = None  # Set to a file name to enable evaluation, e.g., "my_copywriting_data_eval.jsonl"
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 1
LEARNING_RATE = 2e-4
ADAPTER_SAVE_PATH = "my_final_copywriting_adapter"

# === 2. Data Handling ===
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def download_data():
    """Downloads the training and evaluation data from DigitalOcean Spaces."""
    print("--- [Step 2] Downloading data from DO Spaces... ---")

    # Always download the training file
    s3_train_path = f"s3://{DO_SPACES_BUCKET}/{TRAIN_FILE}"
    try:
        print(f"Downloading {TRAIN_FILE}...")
        subprocess.check_call(["s3cmd", "get", s3_train_path])
        if not os.path.exists(TRAIN_FILE):
            print(f"Error: Training file {TRAIN_FILE} not found locally after download attempt.")
            sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to download training data: {e}")
        sys.exit(1)

    # Conditionally download the evaluation file
    eval_file_path = None
    if EVAL_FILE:
        s3_eval_path = f"s3://{DO_SPACES_BUCKET}/{EVAL_FILE}"
        try:
            print(f"Downloading {EVAL_FILE}...")
            subprocess.check_call(["s3cmd", "get", s3_eval_path])
            if os.path.exists(EVAL_FILE):
                eval_file_path = EVAL_FILE
            else:
                print(f"Warning: Evaluation file {EVAL_FILE} not found after download attempt. Proceeding without evaluation.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not download evaluation file {EVAL_FILE}: {e}. Proceeding without evaluation.")

    print("--- Data download process complete. ---")
    return TRAIN_FILE, eval_file_path

def format_prompt(example, tokenizer):
    """Formats a data example into the Alpaca prompt format."""
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    formatted_prompt = alpaca_prompt.format(instruction, output)
    return {'text': formatted_prompt + tokenizer.eos_token}

def main():
    """Main function to orchestrate the fine-tuning process."""
    setup_environment()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    import torch

    # Model and LoRA Initialization
    print("--- [Step 3] Loading Model... ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    train_file, eval_file = download_data()

    train_dataset = load_dataset("json", data_files=train_file, split="train")
    train_dataset = train_dataset.map(format_prompt, fn_kwargs={"tokenizer": tokenizer}, batched=False)

    eval_dataset = None
    if eval_file:
        eval_dataset = load_dataset("json", data_files=eval_file, split="train")
        eval_dataset = eval_dataset.map(format_prompt, fn_kwargs={"tokenizer": tokenizer}, batched=False)

    print("--- [Step 4] Configuring LoRA Adapters ---")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=LORA_R,
        use_gradient_checkpointing='unsloth',
    )

    training_arguments = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_steps=20,
        evaluation_strategy="steps" if eval_dataset else "no",
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_arguments,
    )

    print("--- [Step 5] Starting Training... ---")
    trainer.train()

    print("--- Training complete. Saving adapter... ---")
    model.save_pretrained(ADAPTER_SAVE_PATH)
    tokenizer.save_pretrained(ADAPTER_SAVE_PATH)

if __name__ == "__main__":
    main()
