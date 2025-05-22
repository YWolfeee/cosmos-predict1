import os


# export model_name=$1
# export pt_name=$2
# export strategy=$3
# export avg_rate=$4
# export tokenizer_type=$5


# mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_elbo_0.125_nofreeze
# name_list = [
#     "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_elbo_nofreeze",
#     "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_uniform_nofreeze",
#     "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_mse_pure2d_layer88_elboema4safe_nofreeze",
#     "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_static_pure2d_layer88_elbo_nofreeze",
#     "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_static_pure2d_layer88_uniform_nofreeze"
# ]
# for path in name_list:
for rate in [0.75]:
    for method in ['elbo', 'static']:
        for mask in ['mse', 'order4']:
            if method == 'static' and mask == 'mse':
                continue
            path=f"mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_{mask}_pure2d_layer88_{method}_{rate}_nofreeze"

            model_name = path
            pt_name = "iter_000095000"

            os.makedirs(path, exist_ok=True)
            cmd=f"s5cmd --credentials-file ~/.aws/credentials  --profile vfm_checkpoint cp  s3://checkpoints/cosmos_tokenizer2/cosmos/{path}/checkpoints/{pt_name}.pt ./{path}"

            print(cmd)
            os.system(cmd)