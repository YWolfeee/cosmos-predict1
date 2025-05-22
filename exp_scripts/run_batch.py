import os


# export model_name=$1
# export pt_name=$2
# export strategy=$3
# export avg_rate=$4
# export tokenizer_type=$5

# dataset="tokenbench_v1_fps_30_240p_direct_square_clip"
# mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_elbo_0.125_nofreeze
# dataset = 'placeholder'
# name_list = [
#     "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_elbo_nofreeze",
#     # "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_uniform_nofreeze",
#     # "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_mse_pure2d_layer88_elboema4safe_nofreeze",
#     "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_static_pure2d_layer88_elbo_nofreeze",
#     # "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_static_pure2d_layer88_uniform_nofreeze"
# ]
# for rate in [0.25, 0.5, 0.75]:
#     for path in name_list:
#         for method in ['psnr_elbo']:
#             # for dataset in ['tokenbench_v1_fps_30_240p_direct_square', 'tokenbench_v1_fps_30_240p_direct_square_clip']:
#             if 'order4' in path:
#                 tokenizer_type = f"OURS4x8x8-order4-256p-88"
#             elif 'mse' in path:
#                 tokenizer_type = f"OURS4x8x8-mse-256p-88"
#             else:
#                 tokenizer_type = f"OURS4x8x8-256p-88"

#             # if method == 'static' and mask == 'mse':
#             #     continue
#             # path=f"mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_mse_pure2d_layer88_elboema4safe_nofreeze"
            
#             model_name = path
#             pt_name = "iter_000200000_model"
#             strategy = method
#             # strategy = 'global_elbo' if method == 'elbo' else 'static'
#             avg_rate = str(rate)
            

#             cmd = f"sbatch exp_scripts/submit_batch_eval.sh {model_name} {pt_name} {strategy} {avg_rate} {tokenizer_type} {dataset}"

#             print(cmd)
#             os.system(cmd)

# dataset = "davis_square_clip"
for rate in [0.75]:
    for method in ['placeholder']:
        # for mask in ['mse', ]:
        for dataset in ['davis_square_clip', 'tokenbench_v1_fps_30_240p_direct_square_clip']:
            mask = 'mse'
        # for mask in ['mse', 'order4']:
            if method == 'static' and mask == 'mse':
                continue
            # path=f"mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_{mask}_pure2d_layer88_{method}_{rate}_nofreeze"
            path=f"mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_{mask}_pure2d_layer88_elbo_{rate}_nofreeze"

            model_name = path
            pt_name = "iter_000095000"
            # strategy = 'global_elbo' if method == 'elbo' else 'static'
            strategy = method
            avg_rate = str(rate)
            tokenizer_type = f"OURS4x8x8-{mask}-256p-88"

            cmd = f"sbatch exp_scripts/submit_batch_eval.sh {model_name} {pt_name} {strategy} {avg_rate} {tokenizer_type} {dataset}"
            # os.makedirs(path, exist_ok=True)
            # cmd=f"s5cmd --credentials-file ~/.aws/credentials  --profile vfm_checkpoint cp  s3://checkpoints/cosmos_tokenizer2/cosmos/{path}/checkpoints/iter_000042500.pt ./{path}"

            print(cmd)
            os.system(cmd)