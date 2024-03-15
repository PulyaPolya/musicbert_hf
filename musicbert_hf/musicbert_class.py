from transformers import BertConfig, BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import (
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    BertEmbeddings,
    BertEncoder,
    BertOnlyMLMHead,
)
from transformers.utils import (
    add_start_docstrings,
    logging,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)
from transformers.modeling_outputs import MaskedLMOutput
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)


# TODO: (Malcolm 2024-03-13) find the HF equivalent class to fairseq TransformerSentenceEncoder
# TODO: (Malcolm 2024-03-13) do we need to derive this class from something more specific?
# TODO: (Malcolm 2024-03-13) it looks like this functionality is divided between
#   BertEmbeddings and BertEncoder; we mainly want to override the former
class OctupleEncoder(nn.Module):
    def __init__(
        self,
        *args,
        embedding_dim: int,
        compound_ratio: int = 8,
        upsample: bool = True,
        **kwargs,
    ) -> None:
        raise NotImplementedError
        # TODO: (Malcolm 2024-03-13) super?
        super().__init__(*args, **kwargs)
        self.tpu = False
        self.compound_ratio = compound_ratio
        self.downsampling = nn.Sequential(
            nn.Linear(embedding_dim * compound_ratio, embedding_dim)
        )
        self.upsample = upsample
        # (Malcolm 2023-09-07) if `self.upsample` is False, then we shouldn't use
        #   self.upsampling, but we nevertheless create it so that the model will
        #   match the pretrained checkpoints. (We could hack `fairseq` so that
        #   strict=False; there doesn't seem to be a cmd line arg for that.)
        self.upsampling = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * compound_ratio)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: Optional[torch.Tensor] = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]
    ]:
        raise NotImplementedError
        # tokens: batch * compound_seq
        assert tokens.ndim == 2

        batch, compound_seq = tokens.shape
        ratio = self.compound_ratio

        seq = compound_seq / ratio

        assert (
            tokens.shape[1] % ratio == 0
        ), f"token sequences length should be multiple of {ratio} for compound mode"
        # (Malcolm 2024-02-20) I'm not sure what the motivation for this assertion
        #   is, hidden states for intermediate layers seem to work as normal.
        # assert last_state_only, "hidden states not available for compound mode"
        assert positions is None, "custom positions are not supported for compound mode"
        assert (
            token_embeddings is None
        ), "custom token embeddings are not supported for compound mode"
        assert (
            segment_labels is None
        ), "segment embedding not supported for compound mode"

        # # padding mask: boolean tensor (batch, seq)
        # #   where seq = compound_seq // 8
        # padding_mask = tokens[:, ::ratio].eq(self.padding_idx)
        # assert padding_mask.shape == (batch, seq)

        # # TODO: (Malcolm 2024-03-13) have a look at this
        # if not self.traceable and not self.tpu and not padding_mask.any():
        #     # ?
        #     padding_mask = None

        # TODO: (Malcolm 2024-03-13) implement embed_tokens
        # x: batch, compound_seq, embedding_dim
        x = self.embed_tokens(tokens)

        # assert x.shape == (batch, compound_seq, self.embedding_dim)

        # Project from (batch, compound_seq, embedding) -> (batch, seq, ratio * embedding )

        unflattened = x.view(x.shape[0], x.shape[1] // ratio, -1)
        x = self.downsampling(unflattened)

        # assert x.shape == (batch, seq, ratio * self.embedding_dim)

        # Add positional embeddings
        # TODO: (Malcolm 2024-03-13)
        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens[:, ::ratio], positions=positions)

        # done
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        # done
        x = self.dropout_module(x)

        # TODO: (Malcolm 2024-03-13)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)
        if self.upsample:
            x = x.transpose(0, 1)
            x = self.upsampling(x).view(x.shape[0], x.shape[1] * ratio, -1)
            x = x.transpose(0, 1)
        sentence_rep = x[0, :, :]
        if last_state_only:
            inner_states = [x]
        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


# TODO: (Malcolm 2024-03-15) set padding_idx to 1 throughout to match fairseq


class CompoundEmbeddings(BertEmbeddings):
    """
    Closely modeled on `BertEmbeddings` class. We don't subclass because we don't
    want the `word_embeddings` layer.
    """

    def __init__(self, config, *, compound_ratio: int = 8, upsample: bool = True):
        super().__init__(config)
        self.compound_ratio = compound_ratio
        self.downsampling = nn.Sequential(
            nn.Linear(config.hidden_size * compound_ratio, config.hidden_size)
        )
        self.upsample = upsample
        # TODO: (Malcolm 2024-03-13) revise following comment (do we ever use upsampling?)
        # (Malcolm 2023-09-07) if `self.upsample` is False, then we shouldn't use
        #   self.upsampling, but we nevertheless create it so that the model will
        #   match the pretrained checkpoints. (We could hack `fairseq` so that
        #   strict=False; there doesn't seem to be a cmd line arg for that.)
        self.upsampling = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * compound_ratio)
        )

        # Overwrite `position_embeddings` because we need to set padding_idx to match
        #   fairseq implementation
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=1
        )
        # self.position_embeddings = nn.Embedding(
        #     config.max_position_embeddings, config.hidden_size
        # )
        # self.token_type_embeddings = nn.Embedding(
        #     config.type_vocab_size, config.hidden_size
        # )

        # TODO: (Malcolm 2024-03-14) remove I think, based on misunderstanding
        # # Below taken directly from BertEmbeddings

        # # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # # any TensorFlow checkpoint file
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.position_embedding_type = getattr(
        #     config, "position_embedding_type", "absolute"
        # )
        # self.register_buffer(
        #     "position_ids",
        #     torch.arange(config.max_position_embeddings).expand((1, -1)),
        #     persistent=False,
        # )
        # self.register_buffer(
        #     "token_type_ids",
        #     torch.zeros(self.position_ids.size(), dtype=torch.long),
        #     persistent=False,
        # )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # input_ids: batch * compound_seq
        assert input_ids is not None and input_ids.ndim == 2

        batch, compound_seq = input_ids.shape
        ratio = self.compound_ratio

        assert (
            not past_key_values_length
        ), "past_key_values_length not supported for compound mode"
        assert (
            compound_seq % ratio == 0
        ), f"token sequences length should be multiple of {ratio} for compound mode"

        seq = compound_seq // ratio
        # (Malcolm 2024-02-20) I'm not sure what the motivation for this assertion
        #   is, hidden states for intermediate layers seem to work as normal.
        # assert last_state_only, "hidden states not available for compound mode"
        assert (
            position_ids is None
        ), "custom position_ids are not supported for compound mode"

        # (Malcolm 2024-03-13) unlike the fairseq implementation, we give position
        #   ids to padding tokens but I don't think this should matter since we
        #   ignore them later in any case.
        # (Malcolm 2024-03-15) fairseq begins position ids from 2
        position_ids = self.position_ids[:, 2 : seq + 2]

        assert (
            inputs_embeds is None
        ), "inputs_embeds are not supported for compound mode"
        assert (
            token_type_ids is None
        ) or not token_type_ids.any(), (
            "token_type_ids are not supported for compound mode"
        )

        # TODO: (Malcolm 2024-03-13) it seems that the padding mask is not defined
        #   in this module; find where I should define it.
        #   In fact, it seems it is meant to be computed prior to calling BertModel
        #   and provided as the attention_mask argument.

        # padding_mask = input_ids[:, ::ratio].eq(self.padding_idx)
        # assert padding_mask.shape == (batch, seq)

        flat_embeds = self.word_embeddings(input_ids)
        unflattened = flat_embeds.view(
            flat_embeds.shape[0], flat_embeds.shape[1] // ratio, -1
        )
        embeddings = self.downsampling(unflattened)

        # Learned position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def possibly_upsample(self, x):
        if self.upsample:
            x = self.upsampling(x).view(
                x.shape[0], x.shape[1] * self.compound_ratio, -1
            )
        return x


class MusicBertEncoder(BertModel):
    def __init__(self, config, *, compound_ratio: int = 8, add_pooling_layer=False):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config = config
        self.embeddings = CompoundEmbeddings(config)
        # TODO: (Malcolm 2024-03-13) I think we can probably just use the BertEncoder
        #   class without subclassing it
        self.encoder = BertEncoder(config)
        self.compound_ratio = compound_ratio

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if attention_mask is None:
            assert inputs_embeds is None
            assert input_ids is not None

            batch_size, compound_seq = input_ids.shape
            assert (
                compound_seq % self.compound_ratio == 0
            ), f"token sequences length should be multiple of {self.compound_ratio} for compound mode"
            seq = compound_seq // self.compound_ratio
            assert past_key_values is None
            attention_mask = torch.ones(((batch_size, seq)), device=input_ids.device)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            # TODO: (Malcolm 2024-03-13)
            raise NotImplementedError
        else:
            # TODO: (Malcolm 2024-03-13) should this in fact be a dictionary?
            output.last_hidden_state = self.embeddings.possibly_upsample(  # type:ignore
                output.last_hidden_state  # type:ignore
            )
        return output


# @add_start_docstrings(
#     """Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING
# )
class MusicBert(BertPreTrainedModel):
    # TODO: (Malcolm 2024-03-12) _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MusicBert` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = MusicBertEncoder(config, add_pooling_layer=False)
        # TODO: (Malcolm 2024-03-14) verify
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        # TODO: (Malcolm 2024-03-14) verify
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        # TODO: (Malcolm 2024-03-14) verify
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(
    #     BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    # )
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output="'paris'",
    #     expected_loss=0.88,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **model_kwargs
    ):
        raise NotImplementedError
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
