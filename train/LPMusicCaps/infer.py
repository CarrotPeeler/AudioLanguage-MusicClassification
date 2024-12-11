import argparse
import os
import json
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import random
from utils.config import load_config
from model.model_builder import build_bart_model
from dataset.distributed_dataloader import create_dataloader
from tqdm.auto import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal training setup")

    parser.add_argument(
        '--yaml_config_path', 
        type=str, 
        default="train/LPMusicCaps/exp/transfer/music_instruct/mi_short_long_hparams.yaml",
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
    main_worker(config)

def main_worker(config):
    test_loader = create_dataloader(config, split="val")
    
    test_size = len(test_loader.dataset)
    print(f"Train size: {test_size}")
    
    model, device = build_bart_model(config)
    
    eval(config, model, test_loader)
 
def eval(config, model, test_loader):
    save_dir = config.eval.results_save_path
    torch.cuda.set_device(config.gpu_ids[0])
    model.eval()
    
    inference_results = {}
    idx = 0
    for batch in tqdm(test_loader):
        fname, text, audio_tensor, prompt = batch
        if config.gpu_ids[0] != -1:
            audio_tensor = audio_tensor.cuda(config.gpu_ids[0], non_blocking=True)
        with torch.no_grad():
            output = model.generate(
                samples=audio_tensor,
                text_prompt=prompt,
                num_beams=config.eval.num_beams,
                use_nucleus_sampling=config.eval.use_nucleus_sampling,
            )
        for audio_id, gt, pred in zip(fname, text, output):
            inference_results[idx] = {
                "predictions": pred,
                "true_captions": gt,
                "audio_id": audio_id
            }
            idx += 1
    
    with open(os.path.join(save_dir, f"inference.json"), mode="w") as io:
        json.dump(inference_results, io, indent=4)

if __name__ == '__main__':
    main()

    