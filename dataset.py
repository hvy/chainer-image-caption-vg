import numpy as np

from chainer import cuda
from chainer.dataset.convert import to_device
from chainercv.transforms import resize
from chainercv.datasets.visual_genome. \
        visual_genome_region_descriptions_dataset import \
        VisualGenomeRegionDescriptionsDataset
from chainercv.datasets.visual_genome. \
        visual_genome_region_descriptions_dataset import \
        get_vocabulary


_mean_vgg_r = 123.68
_mean_vgg_g = 116.779
_mean_vgg_b = 103.939

_mean_vgg_bgr = np.array([_mean_vgg_b, _mean_vgg_g, _mean_vgg_r],
                         dtype=np.float32).reshape(3, 1, 1)


def get_vg_train():
    train = VisualGenomeRegionDescriptionsDataset()
    return train


def get_vg_vocabulary(include_inverted=False):
    vocab = get_vocabulary()

    if include_inverted:
        ivocab = {token: word for word, token in vocab.items()}
        return vocab, ivocab

    return vocab



def concat_captioning_examples(batch, device=None, padding=None,
                               max_caption_length=None):
    def _concat_arrays(arrays):
        xp = cuda.get_array_module(arrays[0])
        # Add a batch batch axis to each element and concatenate along it
        with cuda.get_device_from_array(arrays[0]):
            return xp.concatenate([array[None] for array in arrays])

    imgs = []
    captions = []
    pad = padding is not None
    for example in batch:
        img, caption = example

        if pad:
            caption_arr = np.empty(max_caption_length, dtype='i')
            caption_arr.fill(padding)
            # Clip to max length if necessary
            caption_arr[:len(caption)] = caption[:max_caption_length]
            caption = caption_arr
        else:
            caption = to_device(device, np.asarray(caption, dtype='i'))

        imgs.append(img)
        captions.append(caption)

    if pad:
        captions = to_device(device, _concat_arrays(captions))
    imgs = to_device(device, _concat_arrays(imgs))

    return imgs, captions


def preprocess_img(img, size=(224, 224), mean_vgg_bgr=_mean_vgg_bgr):
    _, h, w = img.shape
    img = resize(img, size)
    img = img[::-1, : ,:]  # rgb -> bgr
    img -= mean_vgg_bgr
    return img


class ImageCaptionVGG16Transform(object):

    """Contains all code for preprocessing the raw input data.

    Select one caption from the set of all captions for each image and let the
    caption be the caption for the whole image.
    """

    def __init__(self, size=(224, 224), mean_vgg_bgr=_mean_vgg_bgr):
        self.size = size
        self.mean_vgg_bgr = mean_vgg_bgr

    def __call__(self, in_data):
        img, boxes, captions = in_data
        img = preprocess_img(img, self.size, self.mean_vgg_bgr)

        # Select a caption from all
        caption = captions[0]
        return img, caption
