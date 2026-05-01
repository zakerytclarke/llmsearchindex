from huggingface_hub import HfApi, login


api = HfApi()
api.upload_folder(
    folder_path="./models",
    repo_id="zakerytclarke/llmindex",
    repo_type="model",
    commit_message="Optimized memory-mapped assets"
)

print("Upload Complete!")