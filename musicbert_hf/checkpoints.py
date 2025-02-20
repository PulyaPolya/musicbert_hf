import itertools
from typing import Literal, Type, TypeVar

import torch
from transformers import BertConfig, BertPreTrainedModel

from musicbert_hf.models import (
    MusicBert,
    MusicBertMultiTaskTokenClassConditioned,
    MusicBertMultiTaskTokenClassConditionedConfig,
    MusicBertMultiTaskTokenClassification,
    MusicBertTokenClassification,
)
from musicbert_hf.script_helpers.get_vocab import handle_vocab

try:
    import lovely_tensors

    LOVELY_TENSORS_AVAILABLE = True
except ImportError:
    LOVELY_TENSORS_AVAILABLE = False

T = TypeVar("T", bound=BertPreTrainedModel)


def _load_from_checkpoint(
    model_config,
    src_state_dict,
    model_cls: Type[T],
    parameter_mapping: dict | None = None,
    print_missing_keys: bool = False,
    expected_missing_src_keys: list[str] | None = None,
    expected_missing_dst_keys: list[str] | None = None,
    print_state_dicts: bool = False,
    config_cls: Type[BertConfig] = BertConfig,
    **config_kwargs,
) -> T:
    max_positions = model_config.max_positions
    d_model = model_config.encoder_embed_dim
    d_ff = model_config.encoder_ffn_embed_dim
    n_layers = model_config.encoder_layers
    n_heads = model_config.encoder_attention_heads
    padding_idx = model_config.pad
    assert padding_idx == 1

    vocab_size = 1237  # Not sure if there is a way to retrieve this from ckpt_state_dict, can't find it

    bert_config = config_cls(
        num_hidden_layers=n_layers,
        hidden_size=d_model,
        intermediate_size=d_ff,
        vocab_size=vocab_size,
        num_attention_heads=n_heads,
        # 2 seems to be added because 1 is used as padding_idx, so
        #   all position ids must start from 2
        max_position_embeddings=max_positions + 2,
        tie_word_embeddings=False,
        pad_token_id=padding_idx,
        **config_kwargs,
    )

    model = model_cls(bert_config)
    dst_state_dict = model.state_dict()

    if parameter_mapping is None:
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

    if expected_missing_src_keys is None:
        expected_missing_src_keys = []

    if expected_missing_dst_keys is None:
        expected_missing_dst_keys = []

    expected_missing_dst_keys.append("bert.embeddings.token_type_embeddings.weight")

    parameter_keys_not_in_src = []
    parameter_values_not_in_dst = []
    for key, value in parameter_mapping.items():
        if key not in src_state_dict:
            parameter_keys_not_in_src.append(key)
        if value not in dst_state_dict:
            parameter_values_not_in_dst.append(value)

    assert (not parameter_keys_not_in_src) and (not parameter_values_not_in_dst)

    missing_src_keys = []

    remapped_state_dict = {}

    for key, value in src_state_dict.items():
        if key not in parameter_mapping:
            missing_src_keys.append(key)
            if print_missing_keys and key not in expected_missing_src_keys:
                print(f"Source key `{key}` not in `parameter_mapping`")
        else:
            dst_key = parameter_mapping[key]
            remapped_state_dict[dst_key] = value

    missing_dst_keys = []

    for key in dst_state_dict:
        if key not in parameter_mapping.values():
            missing_dst_keys.append(key)
            if print_missing_keys and key not in expected_missing_dst_keys:
                print(f"Dest key `{key}` not in `parameter_mapping`")

    only_in_missing_src_keys = set(missing_src_keys) - set(expected_missing_src_keys)
    only_in_expected_missing_src_keys = set(expected_missing_src_keys) - set(
        missing_src_keys
    )

    only_in_missing_dst_keys = set(missing_dst_keys) - set(expected_missing_dst_keys)
    only_in_expected_missing_dst_keys = set(expected_missing_dst_keys) - set(
        missing_dst_keys
    )
    assert not any(
        [
            only_in_missing_src_keys,
            only_in_expected_missing_src_keys,
            only_in_missing_dst_keys,
            only_in_expected_missing_dst_keys,
        ]
    ), (
        f"{only_in_missing_src_keys=}, {only_in_expected_missing_src_keys=}, {only_in_missing_dst_keys=}, {only_in_expected_missing_dst_keys=}"
    )

    if print_state_dicts:
        print("REMAPPED_STATE_DICT")
        for name, param in remapped_state_dict.items():
            if LOVELY_TENSORS_AVAILABLE:
                print(name, lovely_tensors.lovely(param))
            else:
                print(name, param)
        print("")
        print("HF STATE DICT")
        for name, param in model.state_dict().items():
            if LOVELY_TENSORS_AVAILABLE:
                print(name, lovely_tensors.lovely(param))
            else:
                print(name, param)

    model.load_state_dict(remapped_state_dict, strict=False)
    return model


def load_musicbert_from_fairseq_checkpoint(
    checkpoint_path: str,
    print_missing_keys: bool = False,
    weights_only: bool = False,
) -> MusicBert:
    ckpt_state_dict = torch.load(checkpoint_path, weights_only=weights_only)

    config = ckpt_state_dict["cfg"]
    model_config = config["model"]
    src_state_dict = ckpt_state_dict["model"]
    parameter_mapping = {}

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
    expected_missing_dst_keys = ["cls.predictions.bias"]
    return _load_from_checkpoint(
        model_config,
        src_state_dict,
        model_cls=MusicBert,
        print_missing_keys=print_missing_keys,
        expected_missing_dst_keys=expected_missing_dst_keys,
        parameter_mapping=parameter_mapping,
    )


def load_musicbert_token_classifier_from_fairseq_checkpoint(
    checkpoint_path: str,
    print_missing_keys: bool = False,
    checkpoint_type: Literal["musicbert", "token_classifier"] = "token_classifier",
    num_labels: int | None = None,
    weights_only: bool = False,
    vocab_path: str | None = None,
    **config_kwargs,
) -> MusicBertTokenClassification:
    ckpt_state_dict = torch.load(checkpoint_path, weights_only=weights_only)

    config = ckpt_state_dict["cfg"]
    model_config = config["model"]
    src_state_dict = ckpt_state_dict["model"]

    classifier_dropout = model_config.pooler_dropout
    classifier_activation = model_config.pooler_activation_fn
    expected_missing_src_keys = [
        # The lm_head seems to be in the checkpoint in spite of not being used
        "encoder.lm_head.weight",
        "encoder.lm_head.bias",
        "encoder.lm_head.dense.weight",
        "encoder.lm_head.dense.bias",
        "encoder.lm_head.layer_norm.weight",
        "encoder.lm_head.layer_norm.bias",
    ]
    if checkpoint_type == "musicbert":
        assert num_labels is not None, "num_labels must be provided for musicbert"
        parameter_mapping = {}
        expected_missing_dst_keys = [
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        ]
    elif checkpoint_type == "token_classifier":
        assert num_labels is None, (
            "num_labels must be None for token_classifier (we infer it from the checkpoint)"
        )
        num_labels = src_state_dict[
            "classification_heads.sequence_tagging_head.out_proj.bias"
        ].shape[0]
        parameter_mapping = {
            "classification_heads.sequence_tagging_head.dense.weight": "classifier.dense.weight",
            "classification_heads.sequence_tagging_head.dense.bias": "classifier.dense.bias",
            "classification_heads.sequence_tagging_head.out_proj.weight": "classifier.out_proj.weight",
            "classification_heads.sequence_tagging_head.out_proj.bias": "classifier.out_proj.bias",
        }
        expected_missing_dst_keys = None
    else:
        raise ValueError(f"Invalid checkpoint type: {checkpoint_type}")

    model = _load_from_checkpoint(
        model_config,
        src_state_dict,
        model_cls=MusicBertTokenClassification,  # type:ignore
        print_missing_keys=print_missing_keys,
        parameter_mapping=parameter_mapping,
        expected_missing_src_keys=expected_missing_src_keys,
        expected_missing_dst_keys=expected_missing_dst_keys,
        classifier_dropout=classifier_dropout,
        classifier_activation=classifier_activation,
        num_labels=num_labels,
        **config_kwargs,
    )
    if vocab_path is not None:
        _, stoi = handle_vocab(path=vocab_path)
        model.config.label2id = stoi
        model.config.id2label = {v: k for k, v in stoi.items()}
    return model


def load_musicbert_multitask_token_classifier_from_fairseq_checkpoint(
    checkpoint_path: str,
    print_missing_keys: bool = False,
    checkpoint_type: Literal["musicbert", "token_classifier"] = "token_classifier",
    num_labels: list[int] | None = None,
    weights_only: bool = False,
    vocab_paths: dict[str, str] | None = None,
    **config_kwargs,
) -> MusicBertMultiTaskTokenClassification:
    ckpt_state_dict = torch.load(checkpoint_path, weights_only=weights_only)

    config = ckpt_state_dict["cfg"]
    model_config = config["model"]

    src_state_dict = ckpt_state_dict["model"]
    parameter_mapping = {}

    if checkpoint_type == "musicbert":
        assert num_labels is not None, "num_labels must be provided for musicbert"
        expected_missing_dst_keys = []
        for i in range(len(num_labels)):
            expected_missing_dst_keys.extend(
                [
                    f"classifier.multi_tag_sub_heads.{i}.dense.weight",
                    f"classifier.multi_tag_sub_heads.{i}.dense.bias",
                    f"classifier.multi_tag_sub_heads.{i}.out_proj.weight",
                    f"classifier.multi_tag_sub_heads.{i}.out_proj.bias",
                ]
            )

    elif checkpoint_type == "token_classifier":
        assert num_labels is None, (
            "num_labels must be None for token_classifier (we infer it from the checkpoint)"
        )

        num_labels = []

        if (
            "classification_heads.sequence_multitarget_tagging_head.multi_tag_sub_heads.0.out_proj.bias"
            in src_state_dict
        ):
            multi_label = "multitarget"  # for backwards compatibility
        else:
            multi_label = "multitask"

        for i in itertools.count():
            layer = src_state_dict.get(
                f"classification_heads.sequence_{multi_label}_tagging_head.multi_tag_sub_heads.{i}.out_proj.bias",
                None,
            )
            if layer is None:
                break
            num_labels.append(layer.shape[0])
            parameter_mapping |= {
                f"classification_heads.sequence_{multi_label}_tagging_head.multi_tag_sub_heads.{i}.dense.weight": f"classifier.multi_tag_sub_heads.{i}.dense.weight",
                f"classification_heads.sequence_{multi_label}_tagging_head.multi_tag_sub_heads.{i}.dense.bias": f"classifier.multi_tag_sub_heads.{i}.dense.bias",
                f"classification_heads.sequence_{multi_label}_tagging_head.multi_tag_sub_heads.{i}.out_proj.weight": f"classifier.multi_tag_sub_heads.{i}.out_proj.weight",
                f"classification_heads.sequence_{multi_label}_tagging_head.multi_tag_sub_heads.{i}.out_proj.bias": f"classifier.multi_tag_sub_heads.{i}.out_proj.bias",
            }
        assert num_labels
        expected_missing_dst_keys = []
    classifier_dropout = model_config.pooler_dropout
    classifier_activation = model_config.pooler_activation_fn
    expected_missing_src_keys = [
        # The lm_head seems to be in the checkpoint in spite of not being used
        "encoder.lm_head.weight",
        "encoder.lm_head.bias",
        "encoder.lm_head.dense.weight",
        "encoder.lm_head.dense.bias",
        "encoder.lm_head.layer_norm.weight",
        "encoder.lm_head.layer_norm.bias",
    ]
    model = _load_from_checkpoint(
        model_config,
        src_state_dict,
        model_cls=MusicBertMultiTaskTokenClassification,  # type:ignore
        print_missing_keys=print_missing_keys,
        parameter_mapping=parameter_mapping,
        expected_missing_src_keys=expected_missing_src_keys,
        expected_missing_dst_keys=expected_missing_dst_keys,
        classifier_dropout=classifier_dropout,
        classifier_activation=classifier_activation,
        num_multi_labels=num_labels,
        **config_kwargs,
    )
    if vocab_paths is not None:
        model.config.multitask_label2id = {}
        model.config.multitask_id2label = {}
        for target, vocab_path in vocab_paths.items():
            _, stoi = handle_vocab(path=vocab_path)
            model.config.multitask_label2id[target] = stoi
            model.config.multitask_id2label[target] = {v: k for k, v in stoi.items()}
    return model


def load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint(
    checkpoint_path: str,
    print_missing_keys: bool = False,
    checkpoint_type: Literal["musicbert", "token_classifier"] = "token_classifier",
    num_labels: list[int] | None = None,
    z_vocab_size: int | None = None,
    weights_only: bool = False,
    vocab_paths: dict[str, str] | None = None,
    **config_kwargs,
) -> MusicBertMultiTaskTokenClassConditioned:
    ckpt_state_dict = torch.load(checkpoint_path, weights_only=weights_only)

    config = ckpt_state_dict["cfg"]
    model_config = config["model"]

    src_state_dict = ckpt_state_dict["model"]
    parameter_mapping = {}

    if checkpoint_type == "musicbert":
        assert num_labels is not None, "num_labels must be provided for musicbert"
        assert z_vocab_size is not None, "z_vocab_size must be provided for musicbert"
        expected_missing_dst_keys = []
        for i in range(len(num_labels)):
            expected_missing_dst_keys.extend(
                [
                    f"classifier.multi_tag_sub_heads.{i}.dense.weight",
                    f"classifier.multi_tag_sub_heads.{i}.dense.bias",
                    f"classifier.multi_tag_sub_heads.{i}.out_proj.weight",
                    f"classifier.multi_tag_sub_heads.{i}.out_proj.bias",
                ]
            )
        if "z_mlp_layers" in config_kwargs:
            z_mlp_layers = config_kwargs["z_mlp_layers"]
        else:
            # instantiate default config and retrieve z parameters from it
            config_cls = MusicBertMultiTaskTokenClassConditionedConfig
            config = config_cls(**config_kwargs)
            z_mlp_layers = config.z_mlp_layers
            z_combine_procedure = config.z_combine_procedure
            z_embed_dim = config.z_embed_dim
            z_mlp_norm = config.z_mlp_norm
        for i in range(z_mlp_layers):
            expected_missing_dst_keys.extend(
                [
                    # I'm not actually sure why we have i.0 and i.2 but not i.1
                    f"z_encoder.layers.{i}.0.weight",
                    f"z_encoder.layers.{i}.0.bias",
                    f"z_encoder.layers.{i}.2.weight",
                    f"z_encoder.layers.{i}.2.bias",
                ]
            )
        expected_missing_dst_keys.append("z_encoder.embedding.weight")
    elif checkpoint_type == "token_classifier":
        assert num_labels is None, (
            "num_labels must be None for token_classifier (we infer it from the checkpoint)"
        )
        assert z_vocab_size is None, "z_vocab_size must be None for token_classifier"

        num_labels = []

        z_combine_procedure = model_config.z_combine_procedure
        z_embed_dim = model_config.z_embed_dim
        z_mlp_norm = model_config.z_mlp_norm
        z_vocab_size = src_state_dict["encoder.z_encoder.embedding.weight"].shape[0]

        if (
            "classification_heads.sequence_multitarget_tagging_head.multi_tag_sub_heads.0.out_proj.bias"
            in src_state_dict
        ):
            multi_label = "multitarget"  # for backwards compatibility
        else:
            multi_label = "multitask"

        z_encoder_keys = [key for key in src_state_dict if "z_encoder" in key]
        for key in z_encoder_keys:
            # remove "encoder." prefix
            parameter_mapping[key] = key[8:]

        if any(
            "sequence_multitask_conditional_tagging_head" in key
            for key in src_state_dict
        ):
            chained_output_heads = True
            tagging_head = "conditional_tagging_head"
        else:
            chained_output_heads = False
            tagging_head = "tagging_head"

        for i in itertools.count():
            layer = src_state_dict.get(
                f"classification_heads.sequence_{multi_label}_{tagging_head}.multi_tag_sub_heads.{i}.out_proj.bias",
                None,
            )
            if layer is None:
                break
            num_labels.append(layer.shape[0])
            parameter_mapping |= {
                f"classification_heads.sequence_{multi_label}_{tagging_head}.multi_tag_sub_heads.{i}.dense.weight": f"classifier.multi_tag_sub_heads.{i}.dense.weight",
                f"classification_heads.sequence_{multi_label}_{tagging_head}.multi_tag_sub_heads.{i}.dense.bias": f"classifier.multi_tag_sub_heads.{i}.dense.bias",
                f"classification_heads.sequence_{multi_label}_{tagging_head}.multi_tag_sub_heads.{i}.out_proj.weight": f"classifier.multi_tag_sub_heads.{i}.out_proj.weight",
                f"classification_heads.sequence_{multi_label}_{tagging_head}.multi_tag_sub_heads.{i}.out_proj.bias": f"classifier.multi_tag_sub_heads.{i}.out_proj.bias",
            }
        assert num_labels

        if chained_output_heads:
            for i in range(len(num_labels) - 1):
                parameter_mapping |= {
                    f"classification_heads.sequence_{multi_label}_{tagging_head}.projections.{i}.weight": f"classifier.projections.{i}.weight",
                    f"classification_heads.sequence_{multi_label}_{tagging_head}.projections.{i}.bias": f"classifier.projections.{i}.bias",
                }

        expected_missing_dst_keys = []

    classifier_dropout = model_config.pooler_dropout
    classifier_activation = model_config.pooler_activation_fn
    expected_missing_src_keys = [
        # The lm_head seems to be in the checkpoint in spite of not being used
        "encoder.lm_head.weight",
        "encoder.lm_head.bias",
        "encoder.lm_head.dense.weight",
        "encoder.lm_head.dense.bias",
        "encoder.lm_head.layer_norm.weight",
        "encoder.lm_head.layer_norm.bias",
    ]
    model = _load_from_checkpoint(
        model_config,
        src_state_dict,
        model_cls=MusicBertMultiTaskTokenClassConditioned,  # type:ignore
        print_missing_keys=print_missing_keys,
        parameter_mapping=parameter_mapping,
        expected_missing_src_keys=expected_missing_src_keys,
        expected_missing_dst_keys=expected_missing_dst_keys,
        classifier_dropout=classifier_dropout,
        classifier_activation=classifier_activation,
        num_multi_labels=num_labels,
        z_vocab_size=z_vocab_size,
        z_embed_dim=z_embed_dim,
        z_mlp_norm=z_mlp_norm,
        z_combine_procedure=z_combine_procedure,
        chained_output_heads=chained_output_heads,
        config_cls=MusicBertMultiTaskTokenClassConditionedConfig,
        **config_kwargs,
    )
    if vocab_paths is not None:
        model.config.multitask_label2id = {}
        model.config.multitask_id2label = {}
        for target, vocab_path in vocab_paths.items():
            _, stoi = handle_vocab(path=vocab_path)
            model.config.multitask_label2id[target] = stoi
            model.config.multitask_id2label[target] = {v: k for k, v in stoi.items()}
    return model
