export DATASET_DIR="/mnt/rdata8/qy_dataset/TokenBench"
export GT_VIDEO_DIR="testset_diverse_fps_30_240p"
export GT_VIDEO_CLIPS_DIR="testset_diverse_fps_25_240p_clip"
export OUTPUT_VIDEO_CLIPS_DIR="reconstruction_testset_diverse_fps_25_240p_clip"
export CHECKPOINT_DIR="checkpoints"

# # Special + Layer (8, 8) + Elbo
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_speicial2d_layer88_elboema_nofreeze"
# export pt_name="iter_000055000"
# export strategy="elbo"
# export avg_rate=0.5
# export tokenizer_type="OURS4x8x8-special-256p-88"

# Special + Layer (8, 8) + Static
export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_speicial2d_layer88_static_nofreeze"
export pt_name="iter_000130000"
export strategy="static"
export avg_rate=1.0
export tokenizer_type="OURS4x8x8-special-256p-88"

# # Pure2D + Layer (8, 8) + Elbo
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_pure2d_layer88_elbo_nofreeze"
# export pt_name="iter_000150000"
# export strategy="global_elbo"
# export avg_rate=0.5
# export tokenizer_type="OURS4x8x8-256p-88"

# # Pure2D + Layer (8, 8) + Uniform
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_pure2d_layer88_uniform_nofreeze"
# export pt_name="iter_000150000"
# export strategy="elbo"
# export avg_rate=0.7
# export tokenizer_type="OURS4x8x8-256p-88"

# export model_name="Cosmos-Tokenize1-DV4x8x8-360p"
# export pt_name="model"
# export strategy="static"
# export avg_rate=1.0
# export tokenizer_type="DV4x8x8-360p"
