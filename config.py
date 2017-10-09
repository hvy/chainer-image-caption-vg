import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-iters', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--rnn', type=str, default='lstm',
                        choices=['lstm', 'nsteplstm'])
    parser.add_argument('--ignore-label', type=int, default=-1)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)

    parser.add_argument('--log-iter', type=int, default=1)
    parser.add_argument('--snapshot-iter', type=int, default=10000)

    # Test specific
    parser.add_argument('--test-model', type=str, default='model_10000')
    parser.add_argument('--img-dir', type=str)
    parser.add_argument('--img', type=str)


    return parser.parse_args()
