import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--para_config', type=str, default='./config2.yaml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--unlearn_task', type=str, default='node')
    parser.add_argument('--unlearn_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default="Adam", help='Adam or SGD')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--method', type=str, default="ETR")
    parser.add_argument('--hops', type=int, default=2, help='gnn layers')
    parser.add_argument('--target_model', type=str, default='GCN')
    parser.add_argument('--dim', type=int, default=256)

    args = vars(parser.parse_args())

    return args
