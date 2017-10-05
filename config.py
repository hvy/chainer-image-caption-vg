import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-epoch', default=30)
    parser.add_argument('--batch-size', default=128)
    parser.add_argument('--out', default='result_2')
    parser.add_argument('--rnn', default='lstm', choices=['lstm', 'nsteplstm'])

    # Snapshot intervals
    parser.add_argument('--iter-log', default=1)
    parser.add_argument('--iter-plot', default=1)
    parser.add_argument('--iter-model-snapshot', default=5000)

    # Test specific
    parser.add_argument('--test-model', default='model_1000')
    parser.add_argument('--image', type=str)


    return parser.parse_args()
