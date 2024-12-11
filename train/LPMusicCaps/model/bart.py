import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import AudioEncoder
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

class BartCaptionModel(nn.Module):
    def __init__(self, n_mels=128, num_of_conv=6, sr=16000, duration=10, max_length=128, label_smoothing=0.1, bart_type="facebook/bart-base", audio_dim=768):
        super(BartCaptionModel, self).__init__()
        # non-finetunning case
        bart_config = BartConfig.from_pretrained(bart_type)
        self.tokenizer = BartTokenizer.from_pretrained(bart_type)
        self.bart = BartForConditionalGeneration(bart_config)
        
        self.n_sample = sr * duration
        self.hop_length = int(0.01 * sr) # hard coding hop_size
        self.n_frames = int(self.n_sample // self.hop_length)
        self.num_of_stride_conv = num_of_conv - 1
        self.n_ctx = int(self.n_frames // 2**self.num_of_stride_conv) + 1
        self.audio_encoder = AudioEncoder(
            n_mels = n_mels, # hard coding n_mel
            n_ctx = self.n_ctx, 
            audio_dim = audio_dim, 
            text_dim = self.bart.config.hidden_size,
            num_of_stride_conv = self.num_of_stride_conv
        )

        self.max_length = max_length
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing= label_smoothing, ignore_index=-100)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.ls
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids
    
    def forward_encoder(self, audio, prompt=None):
        # Get audio embeddings
        audio_embs = self.audio_encoder(audio)

        # If a prompt is provided, embed it
        if prompt:
            prompt_tokenized = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            prompt_input_ids = prompt_tokenized["input_ids"].to(self.device)
            prompt_attention_mask = prompt_tokenized["attention_mask"].to(self.device)
            
            # Embed the prompt tokens
            prompt_embeds = self.bart.model.encoder.embed_tokens(prompt_input_ids)

            # Concatenate prompt embeddings with audio embeddings
            encoder_inputs_embeds = torch.cat((prompt_embeds, audio_embs), dim=1)

            # Extend the attention mask accordingly
            attention_mask = torch.cat((prompt_attention_mask, torch.ones(audio_embs.size(0), audio_embs.size(1)).to(self.device)), dim=1)
        else:
            encoder_inputs_embeds = audio_embs
            attention_mask = torch.ones(audio_embs.size(0), audio_embs.size(1)).to(self.device)

        # Pass to encoder
        encoder_outputs = self.bart.model.encoder(
            input_ids=None,
            inputs_embeds=encoder_inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )["last_hidden_state"]

        return encoder_outputs, audio_embs

    def forward_decoder(self, text, encoder_outputs):
        text = self.tokenizer(text,
                              text_pair=None,
                              padding='longest',
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_input_ids = self.shift_tokens_right(
            decoder_targets, self.bart.config.pad_token_id, self.bart.config.decoder_start_token_id
        )

        decoder_outputs = self.bart(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
            encoder_outputs=(encoder_outputs,),
            return_dict=True
        )
        lm_logits = decoder_outputs["logits"]
        loss = self.loss_fct(lm_logits.view(-1, self.tokenizer.vocab_size), decoder_targets.view(-1))
        return loss

    def forward(self, audio, text, prompt=None):
        # format text batch correctly
        if isinstance(text, list):
            if all(isinstance(item, list) for item in text):
                text = text[0]
        if isinstance(prompt, list):
            if all(isinstance(item, list) for item in prompt):
                prompt = prompt[0]
        encoder_outputs, _ = self.forward_encoder(audio, prompt)
        loss = self.forward_decoder(text, encoder_outputs)
        return loss
    
    def generate(self,
                samples,
                text_prompt=None,
                use_nucleus_sampling=False,
                num_beams=5,
                max_length=128,
                min_length=2,
                top_p=0.9,
                repetition_penalty=1.0):
        
        # Encode audio features
        audio_embs = self.audio_encoder(samples)
        
        # Encode text prompt if provided
        if text_prompt:
            prompt_tokens = self.tokenizer(
                text_prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_length,  
                return_tensors="pt"
            ).to(self.device)
            
            prompt_embs = self.bart.model.encoder.embed_tokens(prompt_tokens["input_ids"])
            prompt_embs = prompt_embs.repeat(audio_embs.size(0), 1, 1)  # Shape becomes [3, 11, 768]
            combined_embs = torch.cat((prompt_embs, audio_embs), dim=1)
            
            # Extend attention mask to match audio_embs' batch size
            prompt_attention_mask = prompt_tokens["attention_mask"].repeat(audio_embs.size(0), 1)  # [3, 11]

            # Create audio attention mask
            audio_attention_mask = torch.ones(audio_embs.size(0), audio_embs.size(1)).to(self.device)  # [3, 32]

            # Concatenate the attention masks along the sequence dimension (dim=1)
            attention_mask = torch.cat((prompt_attention_mask, audio_attention_mask), dim=1)  # [3, 43]
        else:
            combined_embs = audio_embs
            attention_mask = None  # No prompt, only use audio
            
        # Pass through encoder
        encoder_outputs = self.bart.model.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=combined_embs,
            return_dict=True
        )
        
        # Prepare decoder start tokens
        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
        input_ids[:, 0] = self.bart.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
        
        # Use the appropriate sampling strategy for generation
        if use_nucleus_sampling:
            outputs = self.bart.generate(
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1
            )
        else:
            outputs = self.bart.generate(
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty
            )
        
        # Decode generated sequences into text
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions