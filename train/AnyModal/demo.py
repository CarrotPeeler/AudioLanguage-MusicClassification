import torch
import torch.nn as nn
import argparse 
import random
from utils.config import load_config
from dataset.cmi_dataset import CMIDataset
from omegaconf import OmegaConf
from models import audio, llm, anymodal

class MultiModalModel(nn.Module):
    """
    MultiModalModel: A flexible architecture for encoding non-text inputs and generating text outputs.
    
    This model supports a variety of input modalities through a customizable input processor and encoder.
    It integrates these modalities with a language model for text generation tasks.
    """

    def __init__(
            self,
            input_processor,
            input_encoder,
            input_tokenizer,
            language_tokenizer,
            language_model,
            config,
        ):
        """
        Initializes the MultiModalModel.
        
        Parameters:
        - input_processor: Callable, processes raw input data.
        - input_encoder: nn.Module, encodes processed input into a latent representation.
        - input_tokenizer: nn.Module, maps encoded input to token embeddings.
        - language_tokenizer: Tokenizer, converts text to tokens for the language model.
        - language_model: nn.Module, pre-trained language model for text generation.
        - input_start_token: str, special token marking the start of input.
        - input_end_token: str, special token marking the end of input.
        """
        super().__init__()
        
        # set current device and transfer model to it
        if len(config.gpu_ids) > 1:
            self.device = torch.cuda.current_device()
        elif len(config.gpu_ids) == 1:
            self.device = config.gpu_ids[0]
        else:
            self.device = "cpu"

        self.config = config

        # Model components
        self.input_processor = input_processor
        self.input_encoder = input_encoder.to(self.device)
        self.input_tokenizer = input_tokenizer.to(self.device)
        self.language_tokenizer = language_tokenizer
        self.language_model = language_model.to(self.device)

        # system prompt to give model context for task
        self.input_system_prompt = config.model.llm_system_prompt

        # special role tokens
        self.input_user_start_token = self.language_tokenizer.bos_token + "user\n"
        self.input_system_start_token = self.language_tokenizer.bos_token + "system\n"
        self.input_assistant_start_token = self.language_tokenizer.bos_token + "assistant\n"

        # Add special tokens to tokenizer and update embeddings (optional: not needed for AutoTokenizer)
        self._add_special_tokens()

        # Loss function
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100) # ignore index is -100 by default, but we set it explicitly here for clarity!

        # Precompute embeddings for special tokens
        self.start_embedding = self._embed_special_token(self.language_tokenizer.bos_token)
        self.end_embedding = self._embed_special_token(self.language_tokenizer.eos_token + "\n")
        self.start_embedding_user = self._embed_special_token(self.input_user_start_token)
        self.start_embedding_system = self._embed_special_token(self.input_system_start_token)
        self.start_embedding_assistant = self._embed_special_token(self.input_assistant_start_token)
        self.system_prompt_embedding = self._embed_special_token(self.input_system_prompt)

    def _add_special_tokens(self):
        """
        Adds custom tokens to the tokenizer and resizes the language model's embeddings.
        """
        self.language_tokenizer.add_tokens([
            self.input_user_start_token, 
            self.input_system_start_token,
            self.input_assistant_start_token,
            self.language_tokenizer.eos_token,
        ], special_tokens=True)
        self.language_model.resize_token_embeddings(len(self.language_tokenizer))

    def forward(self, batch):
        """
        Performs a forward pass with a batch of input and text data.
        
        Parameters:
        - batch: dict, contains 'input' and 'text'.
        
        Returns:
        - logits: torch.Tensor, model predictions.
        - loss: torch.Tensor, computed loss.
        """
        tokenized_input = self._encode_input(batch['input'])
        text_samples = batch['text'] 
        prompt_samples = batch['instruction']
        batch_size = len(text_samples)

        # add EOS token to all text (groundtruth) input
        for i in range(batch_size):
            text_samples[i] = text_samples[i] + self.language_tokenizer.decode(self.language_model.config.eos_token_id)

        input_embeddings, target_labels, attention_masks = [], [], []
        max_sequence_length = 0

        for i in range(batch_size):   
            # Embed the language prompt
            prompt_embedding = self._embed_special_token(prompt_samples[i])
            # Tokenizing the text sample (groundtruth) for each batch element
            tokenized_text = self.language_tokenizer(text_samples[i], return_tensors="pt")['input_ids'] 
            # Embedding the tokenized text (converting token IDs to actual embeddings)           
            embedded_text = self._embed_tokens(tokenized_text)

            # Combining the different input embeddings into one sequence
            combined_input = torch.cat([
                self.start_embedding.squeeze(0), # A special token or embedding for starting the input sequence
                tokenized_input[i], # The "tokenized" audio input 
                self.end_embedding.squeeze(0), # A special token or embedding for ending the input sequence
                prompt_embedding.squeeze(0), # An embedding representing the "prompt" (e.g., textual instruction)
                embedded_text.squeeze(0) # The groundtruth text response embedding
            ], dim=0)

            combined_input = torch.cat([
                self.start_embedding_system.squeeze(0), 
                self.system_prompt_embedding.squeeze(0), # give model some context for its task
                self.end_embedding.squeeze(0),
                
                self.start_embedding_user.squeeze(0),
                tokenized_input[i], # give model the projected audio features
                self.end_embedding.squeeze(0),

                self.start_embedding_user,
                prompt_embedding.squeeze(0), # give model a question about the audio
                self.end_embedding.squeeze(0),
                
                self.start_embedding_assistant.squeeze(0), # generation prompt to signal model to reply to question, not continue it
                embedded_text.squeeze(0), # The groundtruth answer embedding
                self.end_embedding.squeeze(0),
            ], dim=0)
            
            # Creating a label sequence, which is used for model training
            label_sequence = torch.cat([
                torch.full((combined_input.shape[0] - tokenized_text.size(1),), -100),
                tokenized_text.squeeze(0)
            ], dim=0)

            attention_mask = torch.ones(combined_input.shape[0])
            
            input_embeddings.append(combined_input)
            target_labels.append(label_sequence)
            attention_masks.append(attention_mask)
            max_sequence_length = max(max_sequence_length, combined_input.shape[0])

        # Pad sequences to max length
        for i in range(batch_size):
            pad_length = max_sequence_length - input_embeddings[i].size(0)
            pad_token = torch.full((pad_length,), self.language_model.config.eos_token_id, dtype=torch.long, device=self.device)
            pad_embedding = self._embed_tokens(pad_token)
            input_embeddings[i] = torch.cat([input_embeddings[i], pad_embedding], dim=0)
            target_labels[i] = torch.cat([target_labels[i], torch.full((pad_length,), -100, dtype=torch.long)], dim=0)
            attention_masks[i] = torch.cat([attention_masks[i], torch.zeros(pad_length)], dim=0)

        input_embeddings = torch.stack(input_embeddings).to(self.device)
        target_labels = torch.stack(target_labels).to(self.device)
        attention_masks = torch.stack(attention_masks).to(self.device)

        outputs = self.language_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_masks,
            labels=target_labels
        )

        return outputs.logits, outputs.loss

    @torch.no_grad()
    def generate(self, input_data, max_new_tokens=100, **kwargs):
        """
        Generates text given input data.
        
        Parameters:
        - input_data: dict, raw input data.
        - max_new_tokens: int, maximum tokens to generate.
        
        Returns:
        - str, generated text.
        """
        input_data["input"] = input_data["input"].unsqueeze(0)

        # Embed the language prompt
        prompt_embedding = self._embed_special_token(input_data["instruction"])
        tokenized_input = self._encode_input(input_data["input"])
    
        input_embeddings = torch.cat([
            self.start_embedding_system, 
            self.system_prompt_embedding, # give model some context for its task
            self.end_embedding,
            
            self.start_embedding_user,
            tokenized_input, # give model the projected audio features
            self.end_embedding,

            self.start_embedding_user,
            prompt_embedding.squeeze(0), # give model a question about the audio
            self.end_embedding,
            
            self.start_embedding_assistant, # generation prompt to signal model to reply to question, not continue it
        ], dim=1)

        output_ids = self.language_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=torch.ones(input_embeddings.shape[:2], device=self.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.language_model.config.eos_token_id,
            **kwargs
        )

        return self.language_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _encode_input(self, modality_input):
        """
        Encodes the input modality using the processor, encoder, and tokenizer.
        
        Parameters:
        - modality_input: Raw input data.
        
        Returns:
        - torch.Tensor, tokenized modality input.
        """
        processed_input = self.input_processor(modality_input) if self.input_processor else modality_input
        encoded_input = self.input_encoder(processed_input)
        return self.input_tokenizer(encoded_input).to(self.device)

    def _embed_tokens(self, token_ids):
        """
        Embeds tokenized integers using the language model's embeddings.
        
        Parameters:
        - token_ids: torch.Tensor, tokenized input.
        
        Returns:
        - torch.Tensor, token embeddings.
        """
        return self.language_model.get_input_embeddings()(token_ids.to(self.device))

    def _embed_special_token(self, token):
        """
        Embeds a special token and returns its vector.
        
        Parameters:
        - token: str, special token.
        
        Returns:
        - torch.Tensor, token embedding.
        """
        token_ids = torch.tensor(self.language_tokenizer(token)['input_ids'], device=self.device)
        return self._embed_tokens(token_ids).unsqueeze(0)
    


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal training setup")

    parser.add_argument(
        '--yaml_config_path', 
        type=str, 
        default="train/AnyModal/configs/mert-v1-95m_smollm2-135m-instruct_mi_long.yaml", 
        help='Path to config yaml',
    )

    return parser.parse_args()

if __name__ == '__main__':
    # get args and load config
    args = parse_arguments()
    config = load_config(args.yaml_config_path)

    # Load language model and tokenizer
    llm_tokenizer, llm_model = llm.get_llm(
        config.model.llm_model_id, 
        access_token=config.access_token,
        use_peft=config.train.use_peft,
    )
    llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

    # Load vision model components
    audio_processor, audio_model, audio_hidden_size = audio.get_audio_encoder(config.model.audio_model_id, use_peft=False)

    # Initialize vision tokenizer and encoder
    audio_encoder = audio.AudioEncoder(audio_model)
    audio_tokenizer = audio.Projector(audio_hidden_size, llm_hidden_size, num_hidden=1)

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

    generate_args = OmegaConf.to_container(config.model.llm_generate_params, resolve=True)
    multimodal_model.eval()
    
    # Randomly select samples from the dataset to print
    val_dataset = CMIDataset(
        audio_processor, 
        config.dataset.data_dir,
        config.dataset.annotation_path,
        split="val",
        val_subset_size=5,
    )
    random_indices = random.sample(range(len(val_dataset)), min(config.train.num_print_samples, len(val_dataset)))
    selected_samples = [val_dataset[i] for i in random_indices]

    for sample in selected_samples:
        # Extract data
        ytid = sample['ytid']
        question = sample['instruction']
        ground_truth_answer = sample['text']
        # input_data = sample['input']

        # Generate answer
        generated_answer = multimodal_model.generate(sample, **generate_args)

        # Print results
        print(f"\nYouTube ID: {ytid}")
        print(f"Question: {question}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Ground Truth Answer: {ground_truth_answer}")
