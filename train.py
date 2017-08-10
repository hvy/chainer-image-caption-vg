import matplotlib  # NOQA
matplotlib.use('Agg')  # NOQA

import numpy as np

import chainer
from chainer import optimizers
from chainer import iterators
from chainer import training
from chainer.training import extensions
from chainercv.datasets import TransformDataset
from chainercv.transforms import resize
from chainercv.datasets.visual_genome import  \
    visual_genome_region_descriptions_dataset as VG

import config
from model import ImageCaptionModel


def transform(in_data):
    # Ignore bounding boxes and randomly select one label to use for the whole image
    img, bboxes, labels = in_data
    _, h, w = img.shape

    img = resize(img, (224, 224))
    labels = labels[np.random.randint(0, len(labels))]

    return img, labels


def main(args):
    vocab = VG.get_vocabulary()  # word -> id

    train = VG.VisualGenomeRegionDescriptionsDataset()
    train = TransformDataset(train, transform=transform)
    train_iter = iterators.SerialIterator(train, args.batch_size)  # Batch size is 1 for DenseCap

    model = ImageCaptionModel(vocab=vocab)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    updater = training.updater.StandardUpdater(train_iter,
                                               optimizer=optimizer,
                                               device=args.gpu)

    trainer = training.Trainer(updater, out=args.out, stop_trigger=(args.max_iter, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(args.iter_log, 'iteration')))
    trainer.extend(extensions.PlotReport(['main/loss'], trigger=(args.iter_plot, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss']), trigger=(1, 'iteration'))
    trainer.extend(extensions.snapshot_object(model, 'model_{.updater.iteration}'),
                   trigger=(args.iter_model_snapshot, 'iteration'))
    trainer.extend(extensions.snapshot_object(optimizer, 'optimizer_{.updater.iteration}'),
                   trigger=(args.iter_model_snapshot, 'iteration'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    args = config.parse_args()
    main(args)
