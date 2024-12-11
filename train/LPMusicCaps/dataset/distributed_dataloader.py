import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from .cmi_dataset import CMIDataset

def get_dataset(config):
    return CMIDataset

def create_dataloader(config, split):
    val_subset_size = None

    if split == "train":
        split = "train" if config.train.do_val else "all"
    elif split == "val":
        split = "val"
        val_subset_size = config.train.val_subset_size

    dataset = get_dataset(config)(
        data_dir=config.dataset.data_dir,
        ann_path=config.dataset.annotation_path,
        split=split,
        val_subset_size=val_subset_size,
        inflate=config.dataset.inflate,
    )
    batch_size = int(config.train.batch_size / max(1, len(config.gpu_ids)))
    
    if len(config.gpu_ids) > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if isinstance(sampler, DistributedSampler) else True,
        sampler=sampler,
        num_workers=int(config.num_threads),
        worker_init_fn=loader_worker_init_fn(dataset),
        collate_fn=dataset.collater,
    )
    return data_loader

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if hasattr(loader, "sampler"):
        sampler = loader.sampler
    else:
        raise RuntimeError("Unknown sampler for IterableDataset when shuffling dataset")
    assert isinstance(
        sampler, (WeightedRandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None