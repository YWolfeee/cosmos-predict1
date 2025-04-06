from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.tokenizer.modules import (
    ContinuousFormulation,
    Decoder3DType,
    DecoderType,
    DiscreteQuantizer,
    Encoder3DType,
    EncoderType,
)

from cosmos_predict1.tokenizer.networks.adaptive_discrete_video import AdaptiveDiscreteVideoTokenizer


AdaptiveDiscreteVideoTokenizerConfig: LazyDict = L(AdaptiveDiscreteVideoTokenizer)(
    # The adaptive discrete tokenizer that supports variable-length token sequences
    # - Uses 1D token sequences by flattening 3D tokens
    # - Adaptive token allocation based on rate scores
    # - Supports multiple rate allocation strategies
    # - Discrete quantization with FSQ or LFQ
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="rearrange",
    z_channels=256,
    z_factor=1,
    num_groups=1,
    # Most video tokenizers will use legacy_mode=False for mirrored upsampling
    legacy_mode=False,
    spatial_compression=16,
    temporal_compression=8,
    # Adaptive tokenization parameters
    min_tokens=256,
    max_tokens=2048,
    rate_strategy="elbo",  # Options: uniform, elbo
    # Quantizer parameters 
    quantizer=DiscreteQuantizer.FSQ.name,
    embedding_dim=6,
    levels=[8, 8, 8, 5, 5, 5],
    encoder=Encoder3DType.ViT.name,
    decoder=Decoder3DType.ViT.name,
    name="AdaptiveDiscreteVideoTokenizer",
)