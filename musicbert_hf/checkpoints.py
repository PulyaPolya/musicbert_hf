import torch

from transformers import BertConfig
from musicbert_hf.musicbert_class import MusicBert


def load_from_fairseq_checkpoint(checkpoint_path: str) -> MusicBert:
    ckpt_state_dict = torch.load(checkpoint_path)

    config = ckpt_state_dict["cfg"]
    model_config = config["model"]
    max_positions = model_config.max_positions
    d_model = model_config.encoder_embed_dim
    d_ff = model_config.encoder_ffn_embed_dim
    n_layers = model_config.encoder_layers
    n_heads = model_config.encoder_attention_heads

    vocab_size = 1237  # Not sure if there is a way to retrieve this from ckpt_state_dict, can't find it

    src_state_dict = ckpt_state_dict["model"]

    bert_config = BertConfig(
        num_hidden_layers=n_layers,
        hidden_size=d_model,
        intermediate_size=d_ff,
        vocab_size=vocab_size,
        num_attention_heads=n_heads,
        # 2 seems to be added because 1 is used as padding_idx, so
        #   all position ids must start from 2
        max_position_embeddings=max_positions + 2,
        tie_word_embeddings=False,
    )

    model = MusicBert(bert_config)
    dst_state_dict = model.state_dict()

    parameter_mapping = {}

    for i in range(n_layers):
        for kind in ("key", "value", "query"):
            parameter_mapping[
                f"encoder.sentence_encoder.layers.{i}.self_attn.{kind[0]}_proj.weight"
            ] = f"bert.encoder.layer.{i}.attention.self.{kind}.weight"
            parameter_mapping[
                f"encoder.sentence_encoder.layers.{i}.self_attn.{kind[0]}_proj.bias"
            ] = f"bert.encoder.layer.{i}.attention.self.{kind}.bias"
        parameter_mapping[
            f"encoder.sentence_encoder.layers.{i}.self_attn.out_proj.weight"
        ] = f"bert.encoder.layer.{i}.attention.output.dense.weight"
        parameter_mapping[
            f"encoder.sentence_encoder.layers.{i}.self_attn.out_proj.bias"
        ] = f"bert.encoder.layer.{i}.attention.output.dense.bias"
        parameter_mapping[
            f"encoder.sentence_encoder.layers.{i}.self_attn_layer_norm.weight"
        ] = f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight"
        parameter_mapping[
            f"encoder.sentence_encoder.layers.{i}.self_attn_layer_norm.bias"
        ] = f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias"
        parameter_mapping[f"encoder.sentence_encoder.layers.{i}.fc1.weight"] = (
            f"bert.encoder.layer.{i}.intermediate.dense.weight"
        )
        parameter_mapping[f"encoder.sentence_encoder.layers.{i}.fc1.bias"] = (
            f"bert.encoder.layer.{i}.intermediate.dense.bias"
        )
        parameter_mapping[f"encoder.sentence_encoder.layers.{i}.fc2.weight"] = (
            f"bert.encoder.layer.{i}.output.dense.weight"
        )
        parameter_mapping[f"encoder.sentence_encoder.layers.{i}.fc2.bias"] = (
            f"bert.encoder.layer.{i}.output.dense.bias"
        )
        parameter_mapping[
            f"encoder.sentence_encoder.layers.{i}.final_layer_norm.weight"
        ] = f"bert.encoder.layer.{i}.output.LayerNorm.weight"
        parameter_mapping[
            f"encoder.sentence_encoder.layers.{i}.final_layer_norm.bias"
        ] = f"bert.encoder.layer.{i}.output.LayerNorm.bias"

        # lm_head -> cls.predictions
        parameter_mapping["encoder.lm_head.weight"] = "cls.predictions.decoder.weight"
        parameter_mapping["encoder.lm_head.bias"] = "cls.predictions.decoder.bias"
        # NB HuggingFace model has an extra parameter cls.predictions.bias of same
        #   shape as cls.predictions.decoder.bias. Not sure what that does; it is
        #   initialized to zeros so it shouldn't matter too much but perhaps we can
        #   disable it?
        parameter_mapping["encoder.lm_head.dense.weight"] = (
            "cls.predictions.transform.dense.weight"
        )
        parameter_mapping["encoder.lm_head.dense.bias"] = (
            "cls.predictions.transform.dense.bias"
        )
        parameter_mapping["encoder.lm_head.layer_norm.weight"] = (
            "cls.predictions.transform.LayerNorm.weight"
        )
        parameter_mapping["encoder.lm_head.layer_norm.bias"] = (
            "cls.predictions.transform.LayerNorm.bias"
        )

        # compound embeddings
        parameter_mapping["encoder.sentence_encoder.downsampling.0.weight"] = (
            "bert.embeddings.downsampling.0.weight"
        )
        parameter_mapping["encoder.sentence_encoder.downsampling.0.bias"] = (
            "bert.embeddings.downsampling.0.bias"
        )
        parameter_mapping["encoder.sentence_encoder.upsampling.0.weight"] = (
            "bert.embeddings.upsampling.0.weight"
        )
        parameter_mapping["encoder.sentence_encoder.upsampling.0.bias"] = (
            "bert.embeddings.upsampling.0.bias"
        )
        parameter_mapping["encoder.sentence_encoder.embed_tokens.weight"] = (
            "bert.embeddings.word_embeddings.weight"
        )
        parameter_mapping["encoder.sentence_encoder.embed_positions.weight"] = (
            "bert.embeddings.position_embeddings.weight"
        )
        parameter_mapping["encoder.sentence_encoder.emb_layer_norm.weight"] = (
            "bert.embeddings.LayerNorm.weight"
        )
        parameter_mapping["encoder.sentence_encoder.emb_layer_norm.bias"] = (
            "bert.embeddings.LayerNorm.bias"
        )

    EXPECTED_MISSING_DST_KEYS = [
        "cls.predictions.bias",
        "bert.embeddings.token_type_embeddings.weight",
    ]

    for key, value in parameter_mapping.items():
        assert key in src_state_dict
        assert value in dst_state_dict

    missing_src_keys = []

    remapped_state_dict = {}

    for key, value in src_state_dict.items():
        if key not in parameter_mapping:
            missing_src_keys.append(key)
        else:
            dst_key = parameter_mapping[key]
            remapped_state_dict[dst_key] = value

    assert not missing_src_keys

    missing_dst_keys = []

    for key in dst_state_dict:
        if key not in parameter_mapping.values():
            missing_dst_keys.append(key)

    assert sorted(missing_dst_keys) == sorted(EXPECTED_MISSING_DST_KEYS)

    model.load_state_dict(remapped_state_dict, strict=False)
    return model
