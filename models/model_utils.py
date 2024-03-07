import torch
from torch import nn
import torch.nn.functional as F

from utils import load_pretrained_weights
from utils import device
from models.vision_transformer import vit_small
from models.vision_transformer_extended import vit_small_extended

sigmoid = nn.Sigmoid()


def cosine_sim(embeds, prots):
    prots = prots.unsqueeze(0)
    embeds = embeds.unsqueeze(1)
    return F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30)


def get_pre_trained_vit(ckpt_path):
    model = vit_small(patch_size=16, num_classes=0)
    load_pretrained_weights(model, ckpt_path, 'state_dict', 'vit_small', 16)
    model.to(device)
    return model


def get_vit_with_ssf_model(tuning_depth, ckpt_path, init_ssf, **kwargs):
    model = vit_small_extended(num_classes=0, tuning_mode='ssf', tuning_depth=tuning_depth, init_ssf=init_ssf, **kwargs)
    hidden_dim = 384

    load_pretrained_weights(model, ckpt_path, 'state_dict', 'vit_small', 16)
    model.to(device)

    for name, param in model.named_parameters():
        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name:
            param.requires_grad = False

    return model, hidden_dim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def feature_extraction_and_fusion(model, images, depth):
    """
    utility code for extracting features from a depth of d_f and fusing them
    """
    features = model.get_intermediate_layers(images, depth)
    features = [x[:, 0] for x in features]
    features = torch.cat(features, dim=-1)

    features = F.normalize(features, dim=1)  # l2 normalize the features
    return features
