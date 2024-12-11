"""
AnyModal only provides a multimodal training framework for vision + text
This is our attempt to replicate their vision training pipeline for audio!

Referenced the following for assistance:
https://huggingface.co/m-a-p/MERT-v1-95M
https://huggingface.co/m-a-p/MusiLingo-long-v1 
"""
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from torch import nn
from peft import get_peft_config, get_peft_model, LoraConfig

class Projector(nn.Module):
    """
    Projector: A feedforward neural network for projecting feature embeddings to a target dimension.

    Attributes:
    - inp_layer: Input linear layer.
    - layers: Sequence of hidden layers.
    - dropout: Dropout applied between layers.
    - out_layer: Output linear layer.
    """
    def __init__(self, config, in_features, out_features):
        """
        Initializes the Projector.

        Parameters:
        - in_features: int, size of the input feature vector.
        - out_features: int, size of the output feature vector.
        - num_hidden: int, number of hidden layers (default: 2).
        """
        super(Projector, self).__init__()
        self.config = config
        self.inp_layer = nn.Linear(in_features, out_features)
        # self.layers = nn.ModuleList([
        #     nn.Linear(out_features, out_features) for _ in range(config.model.projector_params.num_hidden)
        # ])
        # self.dropout = nn.Dropout(config.model.projector_params.dropout)
        # self.out_layer = nn.Linear(out_features, out_features)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: torch.Tensor, input tensor.

        Returns:
        - torch.Tensor, output tensor.
        """
        x = self.inp_layer(x)
        # for layer in self.layers:
        #     x = self.dropout(x)
        #     x = layer(x)
        # x = self.out_layer(x)
        return x

class AudioEncoder(nn.Module):
    """
    AudioEncoder: Wraps an audio model to extract hidden states as feature embeddings.
    
    Attributes:
    - model: Pre-trained audio model.
    - device: Torch device (GPU/CPU).
    """
    def __init__(self, config, model):
        """
        Initializes the AudioEncoder.

        Parameters:
        - config: config object
        - model: nn.Module, pre-trained audio model.
        """
        super(AudioEncoder, self).__init__()
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, inputs, attn=None, all_hidden=False):
        """
        Forward pass to obtain feature embeddings.

        Parameters:
        - inputs: dict, preprocessed inputs compatible with the audio model.

        Returns:
        - torch.Tensor, last hidden state of the audio model.
        """
        inputs = inputs.to(self.device)

        if attn:
            outputs = self.model(inputs, output_hidden_states=True, attention_mask=attn)
        else:
            outputs = self.model(inputs, output_hidden_states=True)

        if all_hidden:
            return outputs.hidden_states
        
        return outputs.hidden_states[-1]  # Extract last hidden state

def get_audio_encoder(config):
    """
    Loads an audio model and its processor, optionally applying Parameter-Efficient Fine-Tuning (PEFT).

    ***Only supports Wav2Vec2-based audio encoders at the moment***

    Parameters:
    - config: config object

    Returns:
    - processor: Audio processor for pre-processing.
    - model: Pre-trained audio model.
    - hidden_size: int, size of the model's hidden layer.
    """
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config.model.audio_model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(config.model.audio_model_id, trust_remote_code=True)
    hidden_size = model.config.hidden_size
    
    if config.train.use_peft:
        peft_config = LoraConfig(
            task_type=None, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=['dense']
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad = False

    return processor, model, hidden_size

