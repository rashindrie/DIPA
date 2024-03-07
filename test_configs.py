import argparse
import utils


def get_test_configs():
    parser = argparse.ArgumentParser('DIPA', add_help=False)

    # test args
    parser.add_argument('--test_size', type=int, default=600, metavar='TEST_SIZE',
                        help='The number of test episodes sampled')
    parser.add_argument('--test_type', type=str, choices=['standard', '5shot', '1shot'], default='standard',
                        help="meta-test type, standard varying number of ways and shots as in Meta-Dataset, 1shot for "
                             "five-way-one-shot and 5shot for varying-way-five-shot evaluation.")
    parser.add_argument('--pretrained_setting', type=str, default='MDL', choices=['MDL', 'SDL','SDL_E'],
                        help="The pre-training setting. Will be used to determine the tuning depth depending on "
                             "whether a dataset is in-domain or out-of-domain.")

    parser.add_argument('--max_iter', type=int, default=80, help="The number of fine-tuning iterations")
    parser.add_argument('--feature_depth', type=int, default=4, help="The feature fusion depth")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate for SSF parameters")
    parser.add_argument('--lr_anchors', type=float, default=5.0, help="learning rate for anchors in proxy-anchor")
    parser.add_argument('--test_optimizer', default='nadam', type=str,
                        help="The type of optimizer that is used for fine-tuning. Here, we use the same type of "
                             "optimizer for both SSF and proxy-anchor")
    parser.add_argument('--embedding_loss_type', default='proxy_anchor', type=str,
                        choices=['NCC', 'proxy_anchor', 'proxy_anchor_custom'], help="Choice of fine-tuning objective")
    parser.add_argument('--use_proxy_acc', default=False, type=utils.bool_flag,
                        help="Using proxy-anchor for Query classification. Default is False and we instead use NCC "
                             "for query classification. Morevoer, if using NCC for fine-tuning, NCC will "
                             "automatically be used for query classification as well, regardless of the value of this "
                             "flag.")
    parser.add_argument('--init_ssf', default=False,
                        type=utils.bool_flag, help="Whether SSF parameters have constant or random initializations. "
                                                   "Setting this to True means parameters will have random "
                                                   "initialization while False meanse no randomizations will be "
                                                   "performed.")

    # path args
    parser.add_argument('--checkpoint_path', type=str, default='/checkpoints/checkpoint.pth',
                        help="path to pre-trained feature extractor")
    parser.add_argument('--out_dir', default='./', type=str, metavar='PATH',
                        help='directory to output the result')

    args = vars(parser.parse_args())
    return args
