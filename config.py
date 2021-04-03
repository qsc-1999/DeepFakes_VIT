import argparse

def load_config():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--evaluation_epoch', type=int, default=1)

    return parser.parse_args()