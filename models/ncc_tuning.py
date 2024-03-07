import torch

from models.model_utils import get_parameters
from models.optim_factory import create_optimizer_v2
from models.losses import prototype_loss
from models.model_utils import get_vit_with_ssf_model, feature_extraction_and_fusion


def ncc_tuning(model, context_images, context_labels, feature_depth, max_iter, optimizer_type=None, lr=0.05):
    """
    Fine-tuning using NCC during meta-testing
    """

    optimizer = None

    parameters = get_parameters(model)  # extract the non-frozen parameters in ViT
    if len(parameters):
        # initialize optimizer for non-frozen parameters (SSF) attached to the ViT
        optimizer = create_optimizer_v2(parameters, opt=optimizer_type, lr=lr)

    model.train()

    # fine-tune for max_iter number of iterations using NCC
    for i in range(max_iter):
        if optimizer is not None:
            optimizer.zero_grad()

        context_features = feature_extraction_and_fusion(model, context_images, feature_depth)

        loss, stat, _ = prototype_loss(context_features, context_labels, context_features, context_labels)

        loss.backward()
        if optimizer is not None:
            optimizer.step()

        del context_features

    return model


def run_ncc_tuning(context_images, target_images, context_labels, target_labels, optimizer_type, max_iter,
                   feature_depth, tuning_depth, lr, ckpt_path, init_ssf):
    # load a ViT model with SSF parameters appended to the desired tuning depth
    model_mim, hidden_dim = get_vit_with_ssf_model(tuning_depth=tuning_depth, ckpt_path=ckpt_path, init_ssf=init_ssf)

    if tuning_depth > 0:
        # perform fine-tuning with NCC on support data
        model_mim = ncc_tuning(model_mim, context_images, context_labels, feature_depth,
                                max_iter=max_iter, lr=lr, optimizer_type=optimizer_type)

    # evaluation of fine-tuned model on query set
    model_mim.eval()
    with torch.no_grad():
        context_features = feature_extraction_and_fusion(model_mim, context_images, feature_depth)
        target_features = feature_extraction_and_fusion(model_mim, target_images, feature_depth)

    _, stats_dict, pred = prototype_loss(context_features, context_labels, target_features, target_labels)

    # clearing the mem of unused variables
    del context_features, target_features, model_mim

    return stats_dict
