import argparse
import math
import os
import random
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from dataset.cmi_dataset import CMIDataset
from dataset.distributed_dataloader import create_dataloader, shuffle_dataset
from model.model_builder import build_bart_model
from utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from utils.config import load_config
import utils.distributed as du
import numpy as np
from tqdm.auto import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal training setup")

    parser.add_argument(
        '--yaml_config_path', 
        type=str, 
        default="train/LPMusicCaps/exp/transfer/music_instruct/mi_long_hparams.yaml",
        help='Path to config yaml',
    )
    return parser.parse_args()

def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)  # Ensure deterministic Python hash values
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = True  # if false, may slow down training slightly, but ensures reproducibility

def main():
    args = parse_arguments()
    config = load_config(args.yaml_config_path)
    if config.seed is not None:
        seed_all(config.seed)
    # configure parallel or single GPU train
    du.launch_job(config=config, init_method=config.init_method, func=main_worker)

def main_worker(config):
    # setup distributed
    if len(config.gpu_ids) > 1:
        du.init_distributed_training(len(config.gpu_ids), config.shard_id)

    train_loader = create_dataloader(config, split="train")
    
    train_size = len(train_loader.dataset)
    print(f"Train size: {train_size}")
    
    max_prompt_length, max_text_length = train_loader.dataset.get_max_length()
    print(f"Max question length: {max_prompt_length} | Max answer length {max_text_length}")

    if config.train.checkpoint_path:
        pretrained_object = torch.load(config.train.checkpoint_path, map_location="cpu")
        model_state_dict = pretrained_object['state_dict']
        if config.train.resume:
            optimizer_state_dict = pretrained_object['optimizer']
            config.train.start_epoch = pretrained_object['epoch']
    
    model, device = build_bart_model(config, state_dict=model_state_dict if config.train.checkpoint_path else None)
    optimizer = torch.optim.AdamW(model.parameters(), config.train.lr)

    if config.train.checkpoint_path and config.train.resume:
        optimizer.load_state_dict(optimizer_state_dict)

    save_dir = config.train.model_save_path
    logger = Logger(save_dir)

    # Setup mixed-precision training
    scaler = None
    autocast_dtype = None
    
    if config.train.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16
        print("Using mixed-precision fp16")
    elif config.train.precision == "bf16":
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 doesn't need scaling
        autocast_dtype = torch.bfloat16
        print("Using mixed-precision bf16")
    print(range(config.train.start_epoch, config.train.epochs))
    for epoch in tqdm(range(config.train.start_epoch, config.train.epochs)):
        if len(config.gpu_ids) > 1:
            shuffle_dataset(train_loader, epoch)
            if hasattr(train_loader.dataset, "_set_epoch_num"):
                train_loader.dataset._set_epoch_num(epoch)
        
        # Train the model
        train(train_loader, model, optimizer, epoch, logger, config, scaler, autocast_dtype, device)

        # Free GPU memory
        torch.cuda.empty_cache()

        # Save model only if at specified frequency or last epoch
        if (epoch + 1) % config.train.model_save_freq == 0 or (epoch + 1) == config.train.epochs:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

def train(train_loader, model, optimizer, epoch, logger, config, scaler, autocast_dtype, device):    
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()

    for data_iter_step, batch in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, data_iter_step / iters_per_epoch + epoch, config)
        fname, text, audio_embs, prompt = batch
        if device != "cpu":
            audio_embs = audio_embs.cuda(device, non_blocking=True)
        # compute output
        if autocast_dtype:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                loss = model(audio=audio_embs, text=text, prompt=prompt)
        else:
            loss = model(audio=audio_embs, text=text, prompt=prompt)
        
        train_losses.step(loss.item(), audio_embs.size(0))
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        optimizer.zero_grad()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        if data_iter_step % config.train.print_freq == 0:
            progress.display(data_iter_step)
        
        # sync GPUs after each batch
        torch.cuda.synchronize()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.train.warmup_epochs:
        lr = config.train.lr * epoch / config.train.warmup_epochs 
    else:
        lr = config.train.min_lr + (config.train.lr - config.train.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.train.warmup_epochs) / (config.train.epochs - config.train.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()

    