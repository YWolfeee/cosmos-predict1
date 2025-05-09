from huggingface_hub import login, snapshot_download
import os
import sys
import pathlib

HF_TOKEN = "" # CHANGE THIS

# Adjust path if script is executed from outside util_scripts directory
script_path = pathlib.Path(__file__).resolve()
print("script_path", script_path)
root_dir = script_path.parent.parent if "util_scripts" in str(script_path) else script_path.parent
print("root_dir", root_dir)

login(token=HF_TOKEN, add_to_git_credential=True)
model_name="LanguageBind/Open-Sora-Plan-v1.0.0"
local_dir = f"{root_dir}/pretrained_ckpts/{model_name}"
print("local_dir", local_dir)
os.makedirs(local_dir, exist_ok=True)
print(f"downloading `{model_name}` ...")
snapshot_download(repo_id=f"{model_name}", repo_type="space", local_dir=local_dir)