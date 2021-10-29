import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help="epochs of training")
    parser.add_argument('--party_num', type=int, default=4, help="number of parties: K")
    parser.add_argument('--party_feature_num', type=list, default=[1,4,7,11], help="each party's feature number")
    parser.add_argument('--total_feature_num', type=list, default=23, help="total feature number in dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--budget', type=int, default=12, help="server's budget in incentive mechanism")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--gpu', type=int, default=2, help="GPU ID, -1 for CPU")

    args = parser.parse_args()
    return args