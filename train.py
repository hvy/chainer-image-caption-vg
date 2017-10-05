import matplotlib  # NOQA
matplotlib.use('Agg')  # NOQA

import numpy as np

import chainer
from chainer import optimizers
from chainer import iterators
from chainer import training
from chainer.training import extensions
from chainercv.datasets import TransformDataset

import config
import dataset
from model import ImageCaptionModel



def main(args):
    vocab = dataset.get_vg_vocabulary()
    train = dataset.get_vg_train()
    print('len', len(train))
    train = TransformDataset(train, dataset.ImageCaptionVGG16Transform())
    train_iter = iterators.SerialIterator(train, args.batch_size)  # Batch size is 1 for DenseCap

    model = ImageCaptionModel(vocab=vocab, rnn=args.rnn)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    def concat_captioning_examples(batch, device):
        if args.rnn == 'lstm':
            padding = -1
            max_caption_length = 10
        elif args.rnn == 'nsteplstm':
            padding = None
            max_caption_length = None
        else:
            raise ValueError()
        return dataset.concat_captioning_examples(
            batch, device, padding, max_caption_length)

    updater = training.updater.StandardUpdater(
        train_iter, optimizer=optimizer, device=args.gpu,
        converter=concat_captioning_examples)

    trainer = training.Trainer(updater, out=args.out, stop_trigger=(args.max_epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(args.iter_log, 'iteration')))
    trainer.extend(extensions.PlotReport(['main/loss'], trigger=(args.iter_plot, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'elapsed_time', 'main/loss']), trigger=(1, 'iteration'))
    trainer.extend(extensions.snapshot_object(model, 'model_{.updater.iteration}'),
                   trigger=(args.iter_model_snapshot, 'iteration'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    args = config.parse_args()
    main(args)
