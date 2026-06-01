# from huggingface_hub import snapshot_download
#
# snapshot_download(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     local_dir="./local_models/all-MiniLM-L6-v2",
#     local_dir_use_symlinks=False,
# )



from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    local_dir="./local_models/all-mpnet-base-v2",
)

