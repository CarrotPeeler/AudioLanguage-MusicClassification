import torch
from model.bart import BartCaptionModel
from utils.eval_utils import load_pretrained, print_model_params

def freeze_model(model):
    """
    Freeze the weights of the model by setting requires_grad to False for all parameters.
    
    Args:
        model (nn.Module): The model whose parameters should be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

def build_bart_model(config, state_dict=None):
    model = BartCaptionModel(
        max_length = config.model.max_length,
        label_smoothing = config.model.label_smoothing
    )

    # set current device and transfer model to it
    if len(config.gpu_ids) > 1:
        device = torch.cuda.current_device()
    elif len(config.gpu_ids) == 1:
        device = config.gpu_ids[0]
    else:
        device = "cpu"

    model.to(device)

    # load checkpoint
    if state_dict is not None:
        model.load_state_dict(state_dict)
    print_model_params(model)

    # enable distributed parallel processing if num gpus > 1
    if len(config.gpu_ids) > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=False,
        )
    
    return model, device