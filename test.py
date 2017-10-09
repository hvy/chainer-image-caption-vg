import glob
import numpy as np

import chainer
from chainer import serializers
from chainercv import utils

import config
import dataset
from model import ImageCaptionModel


def main(args):
    if args.img_dir:  # Load all images in directory
        img_paths = [
            i for i in glob.glob(args.img_dir) if i.endswith(('png', 'jpg'))]
        img_paths = sorted(img_paths)
    else:  # Load a single image
        img_paths = [args.img]
    imgs = []
    for img_path in img_paths:
        img = utils.read_image(img_path, color=True)
        img = dataset.preprocess_img(img)
        imgs.append(img)
    imgs = np.asarray(imgs)

    vocab, ivocab = dataset.get_vg_vocabulary(include_inverted=True)
    model = ImageCaptionModel(vocab, rnn=args.rnn,
                              ignore_label=args.ignore_label)
    serializers.load_npz(args.test_model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        imgs = chainer.cuda.to_gpu(imgs)

    with chainer.using_config('train', False):
        captions = model(imgs)
        captions = chainer.cuda.to_cpu(captions)

    for img_path, caption in zip(img_paths, captions):
        caption = ' '.join(ivocab[token] for token in caption) \
            .replace('<bos>', '').replace('<eos>', '').strip()
        print(img_path.split('/')[-1], '\t', caption)


if __name__ == '__main__':
    args = config.parse_args()
    main(args)
