import torch
from . import llm
from . import anymodal
from . import audio

def freeze_model(model):
    """
    Freeze the weights of the model by setting requires_grad to False for all parameters.
    
    Args:
        model (nn.Module): The model whose parameters should be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

def build_multimodal_model(config):
    # Load language model and tokenizer
    llm_tokenizer, llm_model = llm.get_llm(
        config.model.llm_model_id, 
        access_token=config.access_token,
        use_peft=config.train.use_peft,
    )
    llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

    # Load vision model components
    audio_processor, audio_model, audio_hidden_size = audio.get_audio_encoder(config)

    # Initialize vision tokenizer and encoder
    audio_encoder = audio.AudioEncoder(config, audio_model)
    audio_tokenizer = audio.Projector(config, audio_hidden_size, llm_hidden_size)

    # Initialize MultiModalModel
    multimodal_model = anymodal.MultiModalModel(
        input_processor=None,
        input_encoder=audio_encoder,
        input_tokenizer=audio_tokenizer,
        language_tokenizer=llm_tokenizer,
        language_model=llm_model,
        config=config,
    )

    multimodal_model.to(multimodal_model.device)

    # enable distributed parallel processing if num gpus > 1
    if len(config.gpu_ids) > 1:
        # Make model replica operate on the current device
        multimodal_model = torch.nn.parallel.DistributedDataParallel(
            module=multimodal_model,
            device_ids=[multimodal_model.device],
            output_device=multimodal_model.device,
            find_unused_parameters=False,
        )
    
    return audio_processor, multimodal_model