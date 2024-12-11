import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
from typing import Optional
import random
from utils.config import load_config
from utils.schedule import get_scheduler_from_config
from models import anymodal, model_builder
from omegaconf import OmegaConf
import utils.distributed as du
from torch.utils.data import DataLoader
from dataset.distributed_dataloader import create_dataloader, shuffle_dataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def evaluate(
    multimodal_model, 
    val_loader, 
    device, 
    autocast_dtype,  # The desired precision, e.g., torch.float16 or torch.bfloat16
    config,
):
    """
    Validates the model and prints ytid, question, generated answer, and ground truth answer.

    Args: 
        multimodal_model: The question-answering model.
        val_loader: DataLoader for the validation dataset.
        device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
        autocast_dtype: The desired precision for mixed-precision evaluation.
        config: Configuration object with necessary attributes.
    """
    generate_args = OmegaConf.to_container(config.model.llm_generate_params, resolve=True)
    multimodal_model.eval()
    
    # Randomly select samples from the dataset to print
    val_dataset = val_loader.dataset
    random_indices = random.sample(range(len(val_dataset)), min(config.train.num_print_samples, len(val_dataset)))
    selected_samples = [val_dataset[i] for i in random_indices]

    validation_losses = []
    if config.train.do_val:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                logits, loss = multimodal_model(batch)
                validation_losses.append(loss.item())

    if config.train.num_print_samples > 0:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):    
            for sample in selected_samples:
                # Extract data
                ytid = sample['ytid']
                question = sample['instruction']
                ground_truth_answer = sample['text']
                # input_data = sample['input']

                # Generate answer
                if len(config.gpu_ids) > 1: generated_answer = multimodal_model.module.generate(sample, **generate_args)
                else: generated_answer = multimodal_model.generate(sample, **generate_args)

                # Print results
                print(f"\nYouTube ID: {ytid}")
                print(f"Question: {question}")
                print(f"Generated Answer: {generated_answer}")
                print(f"Ground Truth Answer: {ground_truth_answer}")

    return validation_losses

def train_multimodal_audio_qa(
    multimodal_model: anymodal.MultiModalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler, 
    config,
    device: Optional[torch.device] = None,
):
    """
    Trains a multimodal audio question-answering model with optional mixed-precision training.

    Args:
        multimodal_model: The model to train, implementing a callable interface for forward pass.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for updating model parameters.
        config: Config containing training settings, including precision mode.
        device: Optional device to run the training on. If None, defaults to CUDA if available.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    multimodal_model.to(device)

    # Ensure the model save path exists
    os.makedirs(config.train.model_save_path, exist_ok=True)

    # Setup mixed-precision training
    scaler = None
    autocast_dtype = None

    if config.train.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16
    elif config.train.precision == "bf16":
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 doesn't need scaling
        autocast_dtype = torch.bfloat16

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, config.train.epochs + 1):
        if len(config.gpu_ids) > 1:
            shuffle_dataset(train_loader, epoch)
            if hasattr(train_loader.dataset, "_set_epoch_num"):
                train_loader.dataset._set_epoch_num(epoch)

        multimodal_model.train()
        training_losses = []

        print(f"\nEpoch {epoch}/{config.train.epochs}")
        for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            optimizer.zero_grad()

            if autocast_dtype:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    logits, loss = multimodal_model(batch)
            else:
                logits, loss = multimodal_model(batch)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step(epoch, step)

            training_losses.append(loss.item())

            # sync GPUs after each batch
            torch.cuda.synchronize()

        if len(config.gpu_ids) > 1: 
            training_losses = du.all_reduce([torch.tensor(training_losses).to(device)])[0]
            avg_train_loss = training_losses.mean().item()
        else:
            avg_train_loss = sum(training_losses) / len(training_losses)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation
        if epoch % config.train.val_freq == 0:
            validation_losses = evaluate(
                multimodal_model,
                val_loader if config.train.do_val else train_loader, # print samples using train_dataset if not val
                device,
                autocast_dtype,
                config,
            )
            
            if config.train.do_val:
                if len(config.gpu_ids) > 1: 
                    validation_losses = du.all_reduce([torch.tensor(validation_losses).to(device)])[0]
                    avg_val_loss = validation_losses.mean().item()
                else:
                    avg_val_loss = sum(validation_losses) / len(validation_losses)
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Save "best" model checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(config.train.model_save_path, "best_model")
                    if len(config.gpu_ids) > 1: multimodal_model.module._save_model(best_model_path)
                    else: multimodal_model._save_model(best_model_path)
                    print(f"Best model checkpoint saved at: {best_model_path}")

        # Save the model checkpoint
        if epoch % config.train.model_save_freq == 0 or epoch == config.train.epochs:
            save_path = os.path.join(config.train.model_save_path, f"model_epoch_{epoch}")
            if len(config.gpu_ids) > 1: multimodal_model.module._save_model(save_path)
            else: multimodal_model._save_model(save_path)
            print(f"Model checkpoint saved at: {save_path}")

        # in case of fragmented memory
        torch.cuda.empty_cache()

    print("Training complete!")

def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # Ensure deterministic Python hash values
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = True  # if false, may slow down training slightly, but ensures reproducibility

    print(f"Random seed set to: {seed}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal training setup")

    parser.add_argument(
        '--yaml_config_path', 
        type=str, 
        default="train/AnyModal/configs/mert-v1-95m_smollm2-135m-instruct_pretrain1_lpmc.yaml",
        # default="train/AnyModal/configs/mert-v1-95m_smollm2-135m-instruct_finetune1_mi-short.yaml", 
        help='Path to config yaml',
    )

    return parser.parse_args()

def run_train(config):
    """
    Multimodal training pipeline for Audio QA task
    """
    seed_everything(config.random_seed)

    # setup distributed
    if len(config.gpu_ids) > 1:
        du.init_distributed_training(len(config.gpu_ids), config.shard_id)

    audio_processor, multimodal_model = model_builder.build_multimodal_model(config)

    if config.train.checkpoint_path:
        multimodal_model._load_model(config.train.checkpoint_path)

    # load dataset and dataloaders
    train_loader = create_dataloader(config, audio_processor, split="train")
    if config.debug_mode: 
        config.train.val_subset_size = 1000 #2
        train_loader = create_dataloader(config, audio_processor, split="val") # for small subset debugging

    if config.train.do_val:
        val_loader = create_dataloader(config, audio_processor, split="val")

    train_size = len(train_loader.dataset)
    if config.train.do_val: val_size = len(val_loader.dataset)
    print(f"Train size: {train_size}, Validation size: {val_size if config.train.do_val else 0}")

    max_prompt_length, max_text_length = train_loader.dataset.get_max_length()
    print(f"Max question length: {max_prompt_length} | Max answer length {max_text_length}")

    # Training configuration
    device = multimodal_model.device
    multimodal_model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        multimodal_model.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    # scheduler
    scheduler = get_scheduler_from_config(config, optimizer)

    train_multimodal_audio_qa(
        multimodal_model,
        train_loader,
        val_loader if config.train.do_val else None,
        optimizer,
        scheduler,
        config,
        device=device,
    )

if __name__ == '__main__':
    # get args and load config
    args = parse_arguments()
    config = load_config(args.yaml_config_path)

    # configure parallel or single GPU train
    du.launch_job(config=config, init_method=config.init_method, func=run_train)