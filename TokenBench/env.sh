export DATASET_DIR="/joint_training/cosmos-predict1"
export GT_VIDEO_DIR="tokenbench_v1_fps_30_240p"
export GT_VIDEO_CLIPS_DIR="tokenbench_v1_fps_30_240p"
export OUTPUT_VIDEO_CLIPS_DIR="reconstructions_temp"
export CHECKPOINT_DIR="/joint_training/runimage/imaginaire4/ckpts"

export overlap_window=0 # Assign 0 to disable overlapping

# # Pure2d + Layer (8, 8) + Elbo
export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_static_pure2d_layer88_elbo_nofreeze"
export pt_name="iter_000200000"
export strategy="static"
export avg_rate=1.0
export tokenizer_type="OURS4x8x8-256p-88"

# # Special + Layer (8, 8) + Static
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_speicial2d_layer88_static_nofreeze"
# export pt_name="iter_000130000"
# export strategy="static"
# export avg_rate=1.0
# export tokenizer_type="OURS4x8x8-special-256p-88"

# # Pure2D + Layer (8, 8) + MSE + ELBOEMA
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_mse_pure2d_layer88_elboema4safe_nofreeze"
# export pt_name="iter_000090000"
# export strategy="global_elbo"
# export avg_rate=0.5
# export tokenizer_type="OURS4x8x8-mse-256p-88"

# # Pure2D + Layer (8, 8) + ORDER4 + ELBO
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_elbo_0.25_nofreeze"
# export pt_name="iter_000030000"
# export strategy="global_elbo"
# export avg_rate=0.25
# export tokenizer_type="OURS4x8x8-order4-256p-88"

# # Pure2D + Layer (8, 8) + Elbo
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_pure2d_layer88_elbo_nofreeze"
# export pt_name="iter_000150000"
# export strategy="static"
# export avg_rate=1.0
# export tokenizer_type="OURS4x8x8-256p-88"

# # Pure2D + Layer (8, 8) + Uniform
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_pure2d_layer88_uniform_nofreeze"
# export pt_name="iter_000150000"
# export strategy="elbo"
# export avg_rate=0.7
# export tokenizer_type="OURS4x8x8-256p-88"

# DV 4 x 8 x 8
# export model_name="Cosmos-Tokenize1-DV4x8x8-360p"
# export pt_name="model"
# export strategy="static"
# export avg_rate=1.0
# export tokenizer_type="DV4x8x8-360p"

# # DV 8 x 16 x 16
# export model_name="Cosmos-Tokenize1-DV8x16x16-720p"
# export pt_name="model"
# export strategy="static"
# export avg_rate=1.0
# export tokenizer_type="DV8x16x16-720p"


# ############# QY #############

# export DATASET_DIR="/mnt/rdata8/qy_dataset/TokenBench"
# export GT_VIDEO_DIR="tokenbench_v1_fps_30_240p"
# export GT_VIDEO_CLIPS_DIR="subset_v1_fps_30_240p"
# export OUTPUT_VIDEO_CLIPS_DIR="reconstruction_subset_v1_fps_30_240p"
# export CHECKPOINT_DIR="/mnt/rdata8/qy_dataset/TokenBench/checkpoints"

# export overlap_window=17

# # # Special + Layer (8, 8) + Elbo
# # export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_speicial2d_layer88_elboema_nofreeze"
# # export pt_name="iter_000055000"
# # export strategy="elbo"
# # export avg_rate=0.5
# # export tokenizer_type="OURS4x8x8-special-256p-88"

# # # Special + Layer (8, 8) + Static
# # export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_speicial2d_layer88_static_nofreeze"
# # export pt_name="iter_000130000"
# # export strategy="static"
# # export avg_rate=1.0
# # export tokenizer_type="OURS4x8x8-special-256p-88"

# # Pure2D + Layer (8, 8) + MSE + ELBOEMA
# export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_mse_pure2d_layer88_elboema4safe_nofreeze"
# export pt_name="iter_000130000"
# export strategy="static"
# export avg_rate=0.5
# export tokenizer_type="OURS4x8x8-mse-256p-88"

# # # Pure2D + Layer (8, 8) + ORDER4 + ELBO
# # export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_order4_pure2d_layer88_elbo_nofreeze"
# # export pt_name="iter_000017500"
# # export strategy="static"
# # export avg_rate=1.0
# # export tokenizer_type="OURS4x8x8-order4-256p-88"

# # # Pure2D + Layer (8, 8) + Elbo
# # export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_pure2d_layer88_elbo_nofreeze"
# # export pt_name="iter_000150000"
# # export strategy="static"
# # export avg_rate=1.0
# # export tokenizer_type="OURS4x8x8-256p-88"

# # # Pure2D + Layer (8, 8) + Uniform
# # export model_name="mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_pure2d_layer88_uniform_nofreeze"
# # export pt_name="iter_000150000"
# # export strategy="elbo"
# # export avg_rate=0.7
# # export tokenizer_type="OURS4x8x8-256p-88"

# # # DV 4 x 8 x 8
# # export model_name="Cosmos-Tokenize1-DV4x8x8-360p"
# # export pt_name="model"
# # export strategy="static"
# # export avg_rate=1.0
# # export tokenizer_type="DV4x8x8-360p"

# # # DV 8 x 16 x 16
# # export model_name="Cosmos-Tokenize1-DV8x16x16-720p"
# # export pt_name="model"
# # export strategy="static"
# # export avg_rate=1.0
# # export tokenizer_type="DV8x16x16-720p"