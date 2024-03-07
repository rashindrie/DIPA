import torch

from models.model_utils import get_parameters
from models.optim_factory import create_optimizer_v2
from models.losses import get_fine_tuning_loss_function, prototype_loss, cross_entropy_loss
from models.model_utils import get_vit_with_ssf_model, feature_extraction_and_fusion
from models.losses import compute_prototypes
from models.model_utils import get_pre_trained_vit
from utils import device


def proxy_tuning(model, loss_func, context_images, context_labels, feature_depth, max_iter, optimizer_type, lr,
                 lr_proxy):
    """
    Fine-tuning using PA during meta-testing
    """
    optimizer = None

    parameters = get_parameters(model)  # extract the non-frozen parameters in ViT
    if len(parameters) > 0:
        # initialize optimizer for non-frozen parameters (SSF) attached to the ViT
        optimizer = create_optimizer_v2(parameters, opt=optimizer_type, lr=lr)

    # initialize optimizer for proxy-anchor
    optimizer_anchor = create_optimizer_v2(list(loss_func.parameters()), opt=optimizer_type, lr=lr_proxy)

    model.train()
    loss_func.train()

    # fine-tune for max_iter number of iterations using proxy-anchor
    for i in range(max_iter):
        if optimizer is not None:
            optimizer.zero_grad()
        optimizer_anchor.zero_grad()

        context_features = feature_extraction_and_fusion(model, context_images, feature_depth)
        loss = loss_func(context_features, context_labels)
        loss.backward()

        if optimizer is not None:
            optimizer.step()
        optimizer_anchor.step()

        del context_features


def run_proxy_tuning(context_images, target_images, context_labels, target_labels, optimizer_type, max_iter,
                     feature_depth, tuning_depth, lr, lr_proxy, ckpt_path, proxy_acc, embedding_loss_type, init_ssf,
                     ):
    custom_prototypes = None  # required to be defined if the loss type == 'proxy_anchor_custom'

    if embedding_loss_type == "proxy_anchor_custom":
        # if using custom proxy-anchor loss, we need to initialize the prototypes. Here, we first load a
        # pre-trained vit checkpoint without any SSF parameters and use it to obtain feature representations required
        # for prototype calculation.

        model_mim, hidden_dim = get_pre_trained_vit(ckpt_path=ckpt_path)
        with torch.no_grad():
            context_features = feature_extraction_and_fusion(model_mim, context_images, feature_depth)
        custom_prototypes = compute_prototypes(context_features, context_labels, n_way=len(context_labels.unique()))
        del context_features, model_mim

    # load a ViT model with SSF parameters appended to the desired tuning depth
    model_mim, hidden_dim = get_vit_with_ssf_model(tuning_depth=tuning_depth, ckpt_path=ckpt_path, init_ssf=init_ssf)

    # initialize the fine-tuning loss (proxy-anchor or proxy-anchor-custom)
    loss_func = get_fine_tuning_loss_function(context_labels, feature_depth, prototypes=custom_prototypes,
                                              embedding_loss_type=embedding_loss_type, hidden_dim=hidden_dim).to(device)

    # perform fine-tuning with the fine-tuning loss on support data
    proxy_tuning(model_mim, loss_func, context_images, context_labels, feature_depth, max_iter=max_iter, lr=lr,
                 lr_proxy=lr_proxy, optimizer_type=optimizer_type)

    # evaluation of fine-tuned model on query set
    loss_func.eval()
    model_mim.eval()
    with torch.no_grad():
        context_features = feature_extraction_and_fusion(model_mim, context_images, feature_depth)
        target_features = feature_extraction_and_fusion(model_mim, target_images, feature_depth)

    if proxy_acc:
        # if using proxy-anchor for query classification
        logits = loss_func.get_logits(target_features)
        _, stats_dict, pred = cross_entropy_loss(logits, target_labels)
    else:
        # if using NCC for query classification - in our work, the default option is this
        _, stats_dict, pred = prototype_loss(context_features, context_labels, target_features, target_labels)

    # clearing the mem of unused variables
    del loss_func, context_features, target_features

    return stats_dict
