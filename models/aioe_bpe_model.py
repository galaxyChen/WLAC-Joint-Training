# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)
from typing import Optional
from modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
import logging
logger = logging.getLogger(__name__)
import torch.nn as nn
from models.aioe_encoder import AIOEEncoder
from models.aioe_bpe_decoder import AIOEBPEDecoder
from dataclasses import dataclass, field, fields
from typing import List, Optional


@dataclass
class AIOEBPEConfig(TransformerConfig):
    num_mt_decoder_layers: Optional[int] = field(
        default=6, metadata={"help": "number of mt decoder layers in total"}
    )
        

@register_model("aioe_transformer_bpe", dataclass=AIOEBPEConfig)
class AIOEBPEModel(TransformerModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        def spm(path):
            return {
                'path': path,
                'bpe': 'sentencepiece',
                'tokenizer': 'space',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
            'transformer.wmt20.en-ta': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz'),
            'transformer.wmt20.en-iu.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz'),
            'transformer.wmt20.en-iu.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz'),
            'transformer.wmt20.ta-en': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz'),
            'transformer.wmt20.iu-en.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz'),
            'transformer.wmt20.iu-en.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz'),
            'transformer.flores101.mm100.615M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz'),
            'transformer.flores101.mm100.175M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        cfg = AIOEBPEConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
        self.encoder.embed_positions = SinusoidalPositionalEmbedding(embedding_dim=self.encoder.embed_positions.embedding_dim, padding_idx=self.encoder.embed_positions.padding_idx, init_size=self.encoder.embed_positions.weights.size(0))
        self.encoder.embed_segments = nn.Embedding( 2, self.encoder.embed_positions.embedding_dim )


    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, AIOEBPEConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = AIOEBPEConfig.from_namespace(args)
        cfg.no_ar_task = task.cfg.no_ar_task
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            AIOEBPEConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        cfg = AIOEBPEConfig.from_namespace(args)
        return AIOEEncoder(
            cfg,
            src_dict,
            embed_tokens
        ) 

    def max_positions(self):
        """Maximum length supported by the model."""
        if getattr(self, "decoder", None) is not None:
            return (self.encoder.max_positions(), self.decoder.max_positions())
        else:
            return (self.encoder.max_positions(), self.encoder.max_positions())

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return AIOEBPEDecoder(
            AIOEBPEConfig.from_namespace(args), tgt_dict, embed_tokens
        )

    # overwrite forward function
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_position_ids,
        src_segment_ids,
        mt_prev_output_tokens=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_position_ids=src_position_ids,
            src_segment_ids=src_segment_ids,
            return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
        if self.training and mt_prev_output_tokens is not None:
            if getattr(self.cfg, "no_ar_task", False) != True:
                mt_decoder_out = self.decoder.mt_decoder(
                    mt_prev_output_tokens,
                    encoder_out=encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
                # we also need encoder_out to get the hidden states of <mask> token
                return decoder_out, mt_decoder_out
            else:
                return decoder_out, None
        else:
            return decoder_out, None

    #TODO: add a new function
    def get_suggestion_targets(self, sample, net_output, suggestion_type):
        """Get targets from either the sample or the net's output."""
        return sample["{}_net_input".format(suggestion_type)]["{}_target".format(suggestion_type)]

# architectures

def _safe_getattr(args, name, default):
    if default is None:
        return getattr(args, name, default)
    else:
        value = getattr(args, name, default)
        if value is None:
            return default
        else:
            return value
        
@register_model_architecture("aioe_transformer_bpe", "aioe_transformer_bpe_base")
def base_architecture(args):
    args.encoder_embed_path = _safe_getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = _safe_getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = _safe_getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = _safe_getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = _safe_getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = _safe_getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = _safe_getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = _safe_getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = _safe_getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = _safe_getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = _safe_getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = _safe_getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = _safe_getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = _safe_getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = _safe_getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = _safe_getattr(args, "activation_dropout", 0.0)
    args.activation_fn = _safe_getattr(args, "activation_fn", "relu")
    args.dropout = _safe_getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = _safe_getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = _safe_getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = _safe_getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = _safe_getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = _safe_getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = _safe_getattr(args, "adaptive_input", False)
    args.no_cross_attention = _safe_getattr(args, "no_cross_attention", False)
    args.cross_self_attention = _safe_getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = _safe_getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = _safe_getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = _safe_getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = _safe_getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = _safe_getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = _safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = _safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = _safe_getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = _safe_getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = _safe_getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = _safe_getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = _safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = _safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = _safe_getattr(args, "quant_noise_scalar", 0)

    args.replace_positon_segment = _safe_getattr(args, "replace_positon_segment", True)
    args.num_mt_decoder_layers = _safe_getattr(args, "num_mt_decoder_layers", 6)

@register_model_architecture("aioe_transformer_bpe", "aioe_transformer_bpe_small")
def small_architecture(args):
    args.decoder_layers = _safe_getattr(args, "decoder_layers", 1)
    base_architecture(args)