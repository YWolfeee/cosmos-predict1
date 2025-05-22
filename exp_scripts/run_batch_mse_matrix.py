import os


# export model_name=$1
# export pt_name=$2
# export strategy=$3
# export avg_rate=$4
# export tokenizer_type=$5

# dataset="tokenbench_v1_fps_30_240p_direct_square"
# mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_order4_pure2d_layer88_elbo_0.125_nofreeze
dataset='placeholder'
for rate in [0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
# for rate in [0.375, 0.625, 0.75, 0.875, 1.0]:
    for method in ['psnr_elbo']:
        # for dataset in ['davis_square_clip']:
        for iteration in [i * 10000 for i in range(17, 18)]:
            mask = 'mse'
            # if method == 'static' and mask == 'mse':
            #     continue
            path=f"mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_mse_pure2d_layer88_elboema4safe_nofreeze"
            
            model_name = path
            pt_name = f"iter_000{iteration}"
            strategy = method
            avg_rate = str(rate)
            tokenizer_type = f"OURS4x8x8-{mask}-256p-88"

            cmd = f"sbatch exp_scripts/submit_batch_eval.sh {model_name} {pt_name} {strategy} {avg_rate} {tokenizer_type} {dataset}"
            # os.makedirs(path, exist_ok=True)
            # cmd=f"s5cmd --credentials-file ~/.aws/credentials  --profile vfm_checkpoint cp  s3://checkpoints/cosmos_tokenizer2/cosmos/{path}/checkpoints/iter_000042500.pt ./{path}"

            print(cmd)
            os.system(cmd)

# for rate in [0.125, 0.25, 0.5]:
#     for method in ['elbo', 'static']:
#         for mask in ['mse', 'order4']:
#             if method == 'static' and mask == 'mse':
#                 continue
#             path=f"mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_{mask}_pure2d_layer88_{method}_{rate}_nofreeze"

#             model_name = path
#             pt_name = "iter_000075000"
#             strategy = 'global_elbo' if method == 'elbo' else 'static'
#             avg_rate = str(rate)
#             tokenizer_type = f"OURS4x8x8-{mask}-256p-88"

#             cmd = f"sbatch exp_scripts/submit_batch_eval.sh {model_name} {pt_name} {strategy} {avg_rate} {tokenizer_type}"
#             # os.makedirs(path, exist_ok=True)
#             # cmd=f"s5cmd --credentials-file ~/.aws/credentials  --profile vfm_checkpoint cp  s3://checkpoints/cosmos_tokenizer2/cosmos/{path}/checkpoints/iter_000042500.pt ./{path}"

#             print(cmd)
#             os.system(cmd)