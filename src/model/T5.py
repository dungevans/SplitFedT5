import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class SplitT5(nn.Module):
    def __init__(self, model_name="philschmid/flan-t5-base-samsum", layer_id=1, **kwargs):
        super().__init__()
        model_name = kwargs.get("model_path", model_name)
        self.layer_id = layer_id

        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.config = full_model.config
        # Silence tie-weights warning for split checkpoints.
        self.config.tie_word_embeddings = False
        self.generation_config = full_model.generation_config

        if self.layer_id == 1:
            self.encoder = full_model.get_encoder()
            self.shared = full_model.shared
        else:
            self.decoder = full_model.get_decoder()
            self.lm_head = full_model.lm_head
            self.shared = full_model.shared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        labels=None,
        decoder_input_ids=None,
        **kwargs,
    ):
        if self.layer_id == 1:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return encoder_outputs.last_hidden_state, attention_mask

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits, None

    # ---- Compatibility hooks for PEFT Seq2Seq wrappers ----
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name=None):
        # PEFT only needs this method to exist on wrapped models.
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_input_name,
        model_kwargs,
        decoder_start_token_id=None,
        bos_token_id=None,
        device=None,
    ):
        if decoder_start_token_id is None:
            decoder_start_token_id = self.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        if device is None:
            device = model_kwargs[model_input_name].device if model_input_name in model_kwargs else torch.device("cpu")
        decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
        return decoder_input_ids, model_kwargs

    def get_encoder(self):
        return self.encoder if self.layer_id == 1 else None

    def get_decoder(self):
        return self.decoder if self.layer_id == 2 else None

    def get_input_embeddings(self):
        return self.shared

    def _shift_right(self, labels):
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        decoder_start_token_id = (
            self.config.decoder_start_token_id if self.config.decoder_start_token_id is not None else pad_token_id
        )

        shifted_ids = labels.new_zeros(labels.shape)
        shifted_ids[..., 1:] = labels[..., :-1].clone()
        shifted_ids[..., 0] = decoder_start_token_id
        shifted_ids.masked_fill_(shifted_ids == -100, pad_token_id)
        return shifted_ids