import argparse

def load_config():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--evaluation_epoch', type=int, default=1)

    return parser.parse_args()