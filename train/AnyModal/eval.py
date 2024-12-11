import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from typing import Callable, List, Dict, Any
from utils.config import load_config
from dataset.cmi_dataset import CMIDataset
from models import anymodal, model_builder
from utils.metrics import get_metrics_from_config
from omegaconf import OmegaConf

def compute_metrics(
    multimodal_model, 
    val_loader, 
    device, 
    metrics: Dict[str, Callable], 
    config,
):
    """
    Computes an arbitrary set of metrics over the validation dataset.

    Args:
        multimodal_model: The multimodal QA model.
        val_loader: DataLoader for the validation set.
        device: Torch device to run evaluation on.
        metrics: Dictionary of metric names to functions (Callable).
        max_new_tokens: Maximum tokens to generate for each sample.

    Returns:
        A dictionary with metric names and their computed values.
    """
    generate_args = OmegaConf.to_container(config.model.llm_generate_params, resolve=True)
    multimodal_model.eval()
    predictions, ground_truths = [], []

    # Generate predictions and collect ground-truths
    print("Generating predictions and collecting ground truths...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            generated_answers = multimodal_model.generate(batch, **generate_args)
            predictions.extend(generated_answers)
            ground_truths.extend(batch["text"])

    # Compute metrics
    print("Computing metrics...")
    results = {}
    for name, metric_fn in metrics.items():
        try:
            results[name] = metric_fn(predictions, ground_truths)
            print(f"{name}: {results[name]:.4f}")
        except Exception as e:
            print(f"Failed to compute {name}: {e}")
            results[name] = None

    return results

def main():
    parser = argparse.ArgumentParser(description="Run evaluation metrics on validation set.")
    parser.add_argument('--yaml_config_path', type=str, required=True, help='Path to config yaml')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.yaml_config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset, dataloader, and model
    audio_processor, multimodal_model = model_builder.build_multimodal_model(config)
    val_dataset = CMIDataset(
        audio_processor, 
        config.dataset.data_dir, 
        config.dataset.annotation_path, 
        split="test"
    )
    multimodal_model._load_model(config.eval.checkpoint_path)

    val_loader = DataLoader(val_dataset, batch_size=config.eval.batch_size, shuffle=False)
    multimodal_model.to(device)

    # Define evaluation metrics
    metrics = get_metrics_from_config(config)

    # Evaluate
    results = compute_metrics(multimodal_model, val_loader, device, metrics, config)

    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}" if value is not None else f"{metric}: Failed")

if __name__ == "__main__":
    main()
