import sys
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from utils import check_dir
from data.meta_dataset_reader import (MetaDatasetEpisodeReader)
from test_configs import get_test_configs
from models.proxy_tuning import run_proxy_tuning
from models.ncc_tuning import run_ncc_tuning


def init_data(pretrained_setting):
    if pretrained_setting in ['SDL', 'SDL_E']:
        train_set = ['ilsvrc_2012']
    else:
        train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']

    validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                      'mscoco']
    test_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                'traffic_sign', 'mscoco', 'mnist', 'cifar10', 'cifar100']

    return train_set, validation_set, test_set


def main():
    args = get_test_configs()

    print(args)
    check_dir(args['out_dir'])

    # load the training, validation and testing sets and create the data-loader
    trainsets, valsets, testsets = init_data(args['pretrained_setting'])
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test_type'])

    TEST_SIZE = args['test_size']
    accs_names = ['ACC']
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False

    feature_depth = args['feature_depth']
    max_iter = args['max_iter']
    lr = args['lr']
    lr_anchors = args['lr_anchors']
    embedding_loss_type = args['embedding_loss_type']
    optimizer_type = args['test_optimizer']
    checkpoint_path = args['checkpoint_path']
    init_ssf = args['init_ssf']
    use_proxy_acc = args['use_proxy_acc']

    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(dataset)

            # we vary the tuning-depth for in-domain and out-of-domain datasets as below
            if dataset in trainsets:
                tuning_depth = 7
            else:
                tuning_depth = 9

            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                # sampling the task's support set and query set for a few-shot classification task. In Meta-Dataset
                # readers, the support set is denoted as 'context set' while query set is denoted as 'target set'.
                sample = test_loader.get_test_task(session, dataset)
                context_images, target_images = sample['context_images'], sample['target_images'],
                context_labels, target_labels = sample['context_labels'], sample['target_labels'],

                if embedding_loss_type in ['NCC']:
                    # using NCC for fine-tuning
                    stats_dict = run_ncc_tuning(context_images, target_images, context_labels, target_labels,
                                               optimizer_type,  max_iter,  feature_depth, tuning_depth,
                                               lr, checkpoint_path,  init_ssf)
                else:
                    # using PA for fine-tuning
                    stats_dict = run_proxy_tuning(context_images, target_images, context_labels, target_labels,
                                                 optimizer_type,  max_iter, feature_depth, tuning_depth,
                                                 lr, lr_anchors, checkpoint_path,  use_proxy_acc, embedding_loss_type,
                                                  init_ssf)

                var_accs[dataset]['ACC'].append(stats_dict['acc'])

                del stats_dict, context_images, target_images

            dataset_acc = np.array(var_accs[dataset]['ACC']) * 100

            print(f"{dataset}: test_acc (ACC) \t {dataset_acc.mean():.2f}%")

    # Print results table - code copied from TSA/URL repository
    print('Results of weighted prototype loss')
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for acc_name in accs_names:
            acc = np.array(var_accs[dataset_name][acc_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out_dir'])
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, 'test-results.npy')
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()


