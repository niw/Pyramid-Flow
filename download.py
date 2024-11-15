from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='rain1011/pyramid-flow-miniflux',
    local_dir='pyramid_flow_model',
    repo_type='model',
    ignore_patterns=[
        'diffusion_transformer_768p/*',
        'diffusion_transformer_image/*'
    ]
)
