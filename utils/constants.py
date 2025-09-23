LOADER_TYPE = {
    "organsmnist": "medmnist", 
    "camelyon17": "camelyon17",
}

EXTENSIONS = {
    "medmnist": ".pkl",
}

N_CLASSES = {
    "organsmnist": 11,
    "camelyon17": 2,
}

EMBEDDING_DIM = {
    "base_patch14_dinov2": 768, #"vit_base_patch14_dinov2.lvd142m"
    "base_patch16_dino": 768, #"vit_base_patch16_224.dino"
    "base_patch16_vit": 768, #"vit_base_patch16_224.augreg_in21k_ft_in1k"
    "small_patch14_dinov2": 384, #"vit_small_patch14_dinov2.lvd142m"
    "small_patch16_dino": 384, #"vit_small_patch16_224.dino"
    "small_patch16_vit": 384, #"vit_small_patch16_224.augreg_in21k_ft_in1k"
}

NUM_WORKERS = 1
