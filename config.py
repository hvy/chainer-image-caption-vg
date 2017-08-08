import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--max-iter', default=10000)
    parser.add_argument('--batch-size', default=128)
    parser.add_argument('--out', default='result_2')

    # Snapshot intervals
    parser.add_argument('--iter-log', default=1)
    parser.add_argument('--iter-plot', default=1)
    parser.add_argument('--iter-model-snapshot', default=1000)

    # Test specific
    parser.add_argument('--test-model', default='model_1000')

    return parser.parse_args()
