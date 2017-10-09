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
    train = TransformDataset(train, dataset.ImageCaptionVGG16Transform())
    train_iter = iterators.SerialIterator(train, args.batch_size)

    model = ImageCaptionModel(vocab, rnn=args.rnn,
                              ignore_label=args.ignore_label,
                              dropout_ratio=args.dropout_ratio)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    def concat_captioning_examples(batch, device):
        if args.rnn == 'lstm':
            padding = -1  # Ignore label for LSTM
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

    trainer = training.Trainer(
        updater, out=args.out, stop_trigger=(args.max_iters, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(args.log_iter, 'iteration')))
    trainer.extend(extensions.PlotReport(
        ['main/loss'], trigger=(args.log_iter, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['elapsed_time', 'epoch', 'iteration', 'main/loss']),
        trigger=(args.log_iter, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_{.updater.iteration}'),
        trigger=(args.snapshot_iter, 'iteration'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    args = config.parse_args()
    main(args)
