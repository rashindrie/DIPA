"""
Code copied and adapted from https://github.com/mrkshllr/FewTURE/tree/main/datasets
"""

import sys

import argparse
import numpy as np
import torch
from tqdm import tqdm
from models.proxy_tuning import run_proxy_tuning
from utils import device, bool_flag
from datasets.cifar_fs import CIFARFS_DatasetLoader
from datasets.miniimagenet import MiniImageNet_DatasetLoader


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexes,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).reshape(-1)
            # no .t() transpose (in contrast to 'permuted' sampler),
            # As such, the data and labels stay in the sequence of order 'aaaaabbbbbcccccdddddeeeee' after reshape,
            # instead of 'abcdeabcdeabcde'...
            yield batch


def run_eval(args):
    lr = args.lr
    lr_anchors = args.lr_anchors
    max_iter = args.max_iter
    test_optimizer = args.test_optimizer
    embedding_loss_type = args.embedding_loss_type
    feature_depth = args.feature_depth
    checkpoint_path = args.checkpoint_path
    init_ssf = args.init_ssf
    use_proxy_acc = args.use_proxy_acc

    # define loaders and tuning depth here, since we evaluate using SDL_E setting, mini-imagenet is considered
    # in-domain and cifar-fs is considered out-of-domain
    if args.dataset == 'mini_imagenet':
        dataset = MiniImageNet_DatasetLoader(args, args.set)
        tuning_depth = 7
    else:
        dataset = CIFARFS_DatasetLoader(args, args.set)
        tuning_depth = 9

    sampler = CategoriesSampler(dataset.label, args.test_size, args.n_way, args.k_shot + args.query)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    tqdm_gen = tqdm(loader)
    print(f"\nEvaluating {args.n_way}-way {args.k_shot}-shot learning scenario with tuning_depth: {tuning_depth}")
    print(f"Using the {args.set} set of {args.dataset} to run evaluation, averaging over "
          f"{args.test_size} episodes.")
    print(f"Data successfully loaded: There are {len(dataset)} images available to sample from.")
    len_tqdm = len(tqdm_gen)

    accs, accs_proxy = [], []

    for i, batch in enumerate(tqdm_gen, 1):
        # sampling the task's support set denoted as 'context set' and query set denoted as 'target set' to be
        # consistent with meta-dataset loaders
        sample = batch[0].view(args.n_way, -1, 3, 224, 224)
        labels = batch[1].numpy()

        unique_targets = np.sort(np.unique(labels))
        target_map = {target: idx for idx, target in enumerate(unique_targets)}
        converted_targets = np.array([target_map[target] for target in labels])
        labels = torch.from_numpy(converted_targets).view(args.n_way, -1)

        context_images = sample[:, :args.k_shot].reshape(-1, 3, 224, 224).to(device)
        context_labels = labels[:, :args.k_shot].flatten().to(device)

        target_images = sample[:, args.k_shot:].reshape(-1, 3, 224, 224).to(device)
        target_labels = labels[:, args.k_shot:].flatten().to(device)

        # using PA for fine-tuning
        stats_dict = run_proxy_tuning(context_images, target_images, context_labels, target_labels,
                                      feature_depth=feature_depth, lr=lr, lr_proxy=lr_anchors,
                                      optimizer_type=test_optimizer, max_iter=max_iter, tuning_depth=tuning_depth,
                                      ckpt_path=checkpoint_path, embedding_loss_type=embedding_loss_type, init_ssf=init_ssf,
                                      proxy_acc=use_proxy_acc
                                      )

        accs.append(stats_dict['acc'])

    acc = np.array(accs) * 100
    mean_acc1 = acc.mean()
    conf1 = (1.96 * acc.std()) / np.sqrt(len(acc))

    print(f"ACC: {mean_acc1:0.1f} +- {conf1:0.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('metatest', add_help=False)
    # test args
    parser.add_argument('--n_way', type=int, default=5, help="The number of classes in the task")
    parser.add_argument('--k_shot', type=int, default=5, help="The number of examples per class in the task")
    parser.add_argument('--query', type=int, default=15, help="The number of query images sampled for each class")
    parser.add_argument('--dataset', type=str, default='cifar_fs', choices=['cifar_fs', 'mini_imagenet'],
                        help="The dataset used for evaluation")
    parser.add_argument('--set', type=str, default='test', help="Whether the evaluation is on train/val/test set. "
                                                                "Default is test set")
    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--test_size', type=int, default=600, help='The number of test episodes sampled')
    parser.add_argument('--max_iter', type=int, default=80, help="The number of fine-tuning iterations")
    parser.add_argument('--feature_feature_depth', type=int, default=4, help="The feature fusion feature_depth")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate for SSF parameters")
    parser.add_argument('--lr_anchors', type=float, default=5.0, help="learning rate for anchors in proxy-anchor")
    parser.add_argument('--test_optimizer', default='nadam', type=str,
                        help="The type of optimizer that is used for fine-tuning. Here, we use the same type of "
                             "optimizer for both SSF and proxy-anchor")
    parser.add_argument('--embedding_loss_type', default='proxy_anchor', type=str,
                        choices=['proxy_anchor'], help="Choice of fine-tuning objective")
    parser.add_argument('--init_ssf', default=False,
                        type=bool_flag, help="Whether SSF parameters have constant or random initializations. "
                                             "Setting this to True means parameters will have random "
                                             "initialization while False meanse no randomizations will be "
                                             "performed.")
    parser.add_argument('--use_proxy_acc', default=False, type=bool_flag,
                        help="Using proxy-anchor for Query classification. Default is False and we instead use NCC "
                             "for query classification. Morevoer, if using NCC for fine-tuning, NCC will "
                             "automatically be used for query classification as well, regardless of the value of this "
                             "flag.")

    # path args
    parser.add_argument('--checkpoint_path', type=str, default='./',
                        help="path to pre-trained checkpoint in SDL_E setting")
    parser.add_argument('--data_path', type=str, default='./', help='path to data folder')
    parser.add_argument('--out_dir', type=str, default='./', help='directory to output the result and checkpoints')

    args = parser.parse_args()

    run_eval(args)
