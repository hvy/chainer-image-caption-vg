import matplotlib  # NOQA
matplotlib.use('Agg')  # NOQA

import os
import six
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from chainer import serializers
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_bbox
from chainercv import utils
from chainercv import transforms

from chainercv.datasets.visual_genome import  \
    visual_genome_region_descriptions_dataset as VG

import config
from model import ImageCaptionModel


def main(args):
    vocab = VG.get_vocabulary()  # word -> id
    ivocab = {v: k for k, v in six.iteritems(vocab)}  # id -> word

    model = ImageCaptionModel(vocab=vocab)
    serializers.load_npz(os.path.join(args.out, args.test_model), model)

    filename = '5622.jpg'
    print('File', filename)
    img = utils.read_image(filename, color=True)
    img = transforms.resize(img, (224, 224))
    img = img[np.newaxis, :]
    print(img.shape)

    candiates = model.predict(img)

    for _, labels, _ in candiates:
        label = [ivocab[word_id] + ' ' for word_id in labels[1:-1]]
        print(label)

    # vis_bbox(img, [[0, 0, 10, 10]], label=[0], label_names=['hello world'])
    # plt.savefig('test_out.png')


if __name__ == '__main__':
    args = config.parse_args()
    main(args)
