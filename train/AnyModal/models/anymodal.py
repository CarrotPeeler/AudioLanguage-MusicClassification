"""
Modified MultiModalModel class, currently only supports SmolLM2 as the LLM model
https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
"""
import torch
import torch.nn as nn
import os

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

        # Add special tokens to tokenizer and update embeddings
        # self.audio_start_token = "<|audio_start|>" 
        # self.audio_end_token = "<|audio_end|>"
        # self._add_special_tokens([self.audio_start_token, self.audio_end_token])

        # Loss function
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100) # ignore index is -100 by default, but we set it explicitly here for clarity!

        # Precompute embeddings for special tokens
        self.audio_start_embedding = self._embed_special_token(self.language_tokenizer.bos_token)
        self.audio_end_embedding = self._embed_special_token(self.language_tokenizer.eos_token + "\n")
        self.system_prompt_embedding = self.embed_message("system", self.input_system_prompt)
        self.generation_prompt_embedding = self.embed_message("assistant", "", add_generation_prompt="only")

    def _add_special_tokens(self, tokens):
        """
        Adds custom tokens to the tokenizer and resizes the language model's embeddings.

        args:
            tokens: list of str
        """
        self.language_tokenizer.add_tokens(tokens, special_tokens=True)
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

        input_embeddings, target_labels, attention_masks = [], [], []
        max_sequence_length = 0

        for i in range(batch_size):   
            # Tokenizing the text sample (groundtruth) for each batch element
            tokenized_text = self.language_tokenizer(text_samples[i], return_tensors="pt")['input_ids'] 

            combined_input = self.apply_chat_template(tokenized_input[i], prompt_samples[i], text_samples[i], mode="train")
            
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
        tokenized_input = self._encode_input(input_data["input"])
    
        input_embeddings = self.apply_chat_template(tokenized_input, input_data["instruction"], mode="test")

        output_ids = self.language_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=torch.ones(input_embeddings.shape[:2], device=self.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.language_model.config.eos_token_id,
            **kwargs
        )

        return self.language_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def tokenize_message(self, role, content, add_generation_prompt=None):
        """
        Tokenize inputs using apply_chat_template()
        args:
            add_generation_prompt: None, "only", or "append"
        """
        # design chat template
        message = [{"role": role, "content": content}]
        template = self.language_tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True if add_generation_prompt == "append" else False,
        )

        # remove system prompt 
        if role != "system": template = template.partition(self.language_tokenizer.eos_token)[-1]
        if add_generation_prompt == "only": template = template.partition(self.language_tokenizer.eos_token)[0]
        
        # tokenize
        input_ids = torch.tensor(self.language_tokenizer(template)['input_ids'])
        return input_ids
    
    def embed_message(self, role, content, add_generation_prompt=False):
        """
        Embeds inputs tokenized using apply_chat_template()
        """
        return self._embed_tokens(self.tokenize_message(role, content, add_generation_prompt))
    
    def apply_chat_template(
            self, 
            tokenized_input, 
            prompt=None,
            text=None,
            mode="train", 
        ):
        """
        Apply chat template formatting for input embeddings
        args:
            mode: "Train" or "Test"
        """
        if self.config.task == "audio-cap":
            if mode == "train":
                # Tokenize and embed the assistant ground truth response (text)
                assistant_embedding = self.embed_message("assistant", text)

                # Concatenate system prompt, tokenized_input, and assistant answer
                template = torch.cat([
                    self.system_prompt_embedding, 
                    self.audio_start_embedding.squeeze(0),
                    tokenized_input, 
                    self.audio_end_embedding.squeeze(0),
                    assistant_embedding,
                ], dim=0)
            else:
                template = torch.cat([
                    self.system_prompt_embedding.unsqueeze(0),
                    self.audio_start_embedding,
                    tokenized_input, # give model the projected audio features
                    self.audio_end_embedding,
                    self.generation_prompt_embedding.unsqueeze(0),
                ], dim=1)
                
        elif self.config.task == "audio-qa":
            if mode == "train":
                # Tokenize and embed prompt
                prompt_embedding = self.embed_message("user", prompt)
                # Tokenize and embed the assistant ground truth
                assistant_embedding = self.embed_message("assistant", text)

                # Concatenate system prompt, tokenized_input, prompt, and assistant answer
                template = torch.cat([
                    self.system_prompt_embedding, 
                    self.audio_start_embedding.squeeze(0),
                    tokenized_input, 
                    self.audio_end_embedding.squeeze(0),
                    prompt_embedding,
                    assistant_embedding,
                ], dim=0)
            else:
                # Tokenize and embed prompt
                prompt_embedding = self.embed_message("user", prompt, add_generation_prompt="append")

                template = torch.cat([
                    self.system_prompt_embedding.unsqueeze(0),
                    self.audio_start_embedding,
                    tokenized_input, # give model the projected audio features
                    self.audio_end_embedding,
                    prompt_embedding.unsqueeze(0),
                ], dim=1)
        else:
            raise TypeError(f"apply_chat_template does not support unknown task: {self.config.task}")
                
        return template

    def _encode_input(self, modality_input, attn=None):
        """
        Encodes the input modality using the processor, encoder, and tokenizer.
        
        Parameters:
        - modality_input: Raw input data.
        
        Returns:
        - torch.Tensor, tokenized modality input.
        """
        # use weighted average of audio encoder's transformer blocks (Inspired by MusiLingo)
        if self.config.model.audio_model_params.use_weighted_avg:
            # Adapted from https://github.com/zihaod/MusiLingo/blob/main/musilingo/models/muvi_model.py

            # both return dims = [25, B, T, 1024]
            if attn is None:
                audio_embeds = torch.stack(self.input_encoder(modality_input, all_hidden=True)) 
            else:
                audio_embeds = torch.stack(self.input_encoder(modality_input, all_hidden=True, attention_mask=attn)) 

            audio_embeds = audio_embeds.transpose(0, 1).mean(-3) #[B, T, 1024]
    
            # Average time steps:
            t = 325
            B, T, D = audio_embeds.shape
            avg_tmp = audio_embeds[:, :T//t*t].reshape(B, T//t, t, D).mean(2)

            # Average the remaining steps
            if T % t > 0:
                avg_last = audio_embeds[:, T//t*t:].reshape(B, 1, T%t, D).mean(2)
                audio_embeds = torch.concat([avg_tmp, avg_last], dim=1)
            else:
                audio_embeds = avg_tmp

            inputs = self.input_tokenizer(audio_embeds).to(self.device)
            return inputs
        else:
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

    def _save_model(self, output_dir, **kwargs):
        """
        Saves the model to disk.
        Implement this method for the functionality to save the model.
        Typically, you only need to save the input tokenizer if you are not training the language model and the input encoder.
        However, if you train peft adapters for input encoder and/or language model, you should consider saving them as well.

        Remember to add the necessary parameters to the function signature as needed.
        
        Parameters:
        - kwargs: Additional arguments for saving.
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.input_tokenizer, f'{output_dir}/input_tokenizer.pt')
    
    def _load_model(self, model_dir, **kwargs):
        """
        Loads the model from disk. Complementary to _save_model.
        Implement this method for the functionality to load the model.
        Remember to add the necessary parameters to the function signature as needed.

        Parameters:
        - kwargs: Additional arguments for loading.
        """
        self.input_tokenizer = torch.load(f'{model_dir}/input_tokenizer.pt')