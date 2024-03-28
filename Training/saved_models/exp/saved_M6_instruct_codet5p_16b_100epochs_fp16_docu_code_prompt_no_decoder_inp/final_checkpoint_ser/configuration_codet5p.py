# coding=utf-8
# Copyright 2023 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.

""" CodeT5+ model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
import copy

logger = logging.get_logger(__name__)


# Adapted from transformers.models.codegen.configuration_codegen.CodeGenConfig
class CodeT5pModuleConfig(PretrainedConfig):
    model_type = "codet5p_module"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size=50400,
            n_positions=2048,
            n_ctx=2048,
            n_embd=4096,
            n_layer=28,
            n_head=16,
            rotary_dim=64,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            tie_word_embeddings=False,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


# Adapted from transformers.models.encoder_decoder.configuration_encoder_decoder.EncoderDecoderConfig
class CodeT5pConfig(PretrainedConfig):
    model_type = "codet5p"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
                "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        encoder_config = kwargs.pop("encoder")
        decoder_config = kwargs.pop("decoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_model_type = decoder_config.pop("model_type")

        if encoder_model_type != decoder_model_type:
            logger.warning("Encoder and decoder model types are different")

        self.encoder = CodeT5pModuleConfig(**encoder_config)
        self.decoder = CodeT5pModuleConfig(**decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
            cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
