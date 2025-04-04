#!/bin/bash



python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m cosmos_predict1.tokenizer.training.train  --config=cosmos_predict1/tokenizer/training/configs/config.py --     experiment=Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA






