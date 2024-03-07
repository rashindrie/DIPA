import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses


def get_pred_stat_dict(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    acc = torch.eq(preds, labels).float().mean()

    stats_dict = {'loss': None, 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}

    return stats_dict, pred_dict


def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')

    stats_dict, pred_dict = get_pred_stat_dict(logits, targets)
    stats_dict['loss'] = loss.item()

    return loss, stats_dict, pred_dict


# NCC
def prototype_loss(support_embeddings, support_labels, query_embeddings, query_labels, distance='cos',
                   return_logits=False, centers=None):

    n_way = len(query_labels.unique())
    if centers is None:
        prots = compute_prototypes(support_embeddings, support_labels, n_way)
    else:
        prots = centers

    prots = prots.unsqueeze(0)
    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)  # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10

    if return_logits:
        return logits, cross_entropy_loss(logits, query_labels)
    return cross_entropy_loss(logits, query_labels)


def compute_prototypes(embeddings, labels, n_way, mean=True):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)

    if mean:
        for i in range(n_way):
            if torch.__version__.startswith('1.1'):
                prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
            else:
                prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    else:
        for i in range(n_way):
            prots[i] = embeddings[(labels == i).nonzero(), :][0]
    return prots


def get_fine_tuning_loss_function(context_labels, feature_depth, prototypes=None, margin=0.1, alpha=32,
                                  embedding_loss_type=None, hidden_dim=0):
    input_dim = int(hidden_dim * feature_depth)
    loss_func = None
    if embedding_loss_type == "proxy_anchor":
        loss_func = losses.ProxyAnchorLoss(
            num_classes=len(context_labels.unique()),
            embedding_size=input_dim,
            margin=margin,
            alpha=alpha,
        )
    elif embedding_loss_type == "proxy_anchor_custom":
        from models.proxy_anchor_custom import ProxyAnchorLoss_Custom
        loss_func = ProxyAnchorLoss_Custom(
            num_classes=len(context_labels.unique()),
            prototypes=prototypes,
            embedding_size=input_dim,
            margin=margin,
            alpha=alpha,
        )
    else:
        print(f"Not implemented: {embedding_loss_type}")

    return loss_func

