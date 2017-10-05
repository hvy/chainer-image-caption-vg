import numpy as np

import chainer
from chainer import functions as F
from chainer import initializers
from chainer import links as L
from chainer import reporter
from chainer import Variable
from chainer import cuda


class ImageFeatureExtractor(chainer.Chain):

    """Image feature extractor.

    Internally use VGG16 or similar CNNs to extract fixed-size features
    from images.
    """

    def __init__(self, cnn, cnn_layer_name):
        super(ImageFeatureExtractor, self).__init__()
        with self.init_scope():
            self.cnn = cnn
        self.cnn_layer_name = cnn_layer_name

    def __call__(self, imgs):
        h = self.cnn(imgs, [self.cnn_layer_name])[self.cnn_layer_name]
        # TODO: Skip the following line or not.
        img_feat = F.dropout(F.relu(h), ratio=0.5)
        return img_feat


class NStepLSTMLanguageModel(chainer.Chain):

    """Recurrent NStepLSTM language model.

    Generate captions given features extracted from images.
    """

    def __init__(self, vocab_size, img_feat_size, hidden_size,
                 dropout_ratio=0.5):
        super(NStepLSTMLanguageModel, self).__init__()
        with self.init_scope():
            self.embed_caption = L.EmbedID(
                vocab_size,
                hidden_size,
                initialW=initializers.Normal(1.0)
            )
            self.embed_img = L.Linear(
                img_feat_size,
                hidden_size,
                initialW=initializers.Normal(0.01),
                initial_bias=initializers.Zero()
            )
            self.lstm = L.NStepLSTM(1, hidden_size, hidden_size, dropout_ratio)
            self.decode_caption = L.Linear(
                hidden_size,
                vocab_size,
                initialW=initializers.Normal(0.01),
                initial_bias=initializers.Zero()
            )

        self.dropout_ratio = dropout_ratio

    def __call__(self, img_feats, captions=None):
        # Initial state based on image features
        h = self.embed_img(img_feats)
        h = F.split_axis(h, h.shape[0], axis=0)
        hx, cx, _ = self.lstm(None, None, h)

        if chainer.config.train:
            caption_lens = [len(c) for c in captions]
            caption_sections = np.cumsum(caption_lens[:-1])
            xs = F.concat(captions, axis=0)
            xs = self.embed_caption(xs)
            xs = F.split_axis(xs, caption_sections, axis=0)

            _, _, ys = self.lstm(hx, cx, xs)

            ys = F.concat(ys, axis=0)
            pred_captions = self.decode_caption(
                F.dropout(ys, self.dropout_ratio))
            ts = F.concat(captions, axis=0)
            loss = F.softmax_cross_entropy(
                pred_captions[:-1], ts[1:], reduce='no')
            loss = F.sum(loss) / (len(ys) - 1)
            return loss

        raise NotImplementedError()


class LSTMLanguageModel(chainer.Chain):

    """Recurrent LSTM language model.

    Generate captions given features extracted from images.
    """

    def __init__(self, vocab_size, img_feat_size, hidden_size,
                 dropout_ratio=0.5, ignore_label=-1):
        super(LSTMLanguageModel, self).__init__()
        with self.init_scope():
            self.embed_word = L.EmbedID(
                vocab_size,
                hidden_size,
                initialW=initializers.Normal(1.0),
                ignore_label=ignore_label
            )
            self.embed_img = L.Linear(
                img_feat_size,
                hidden_size,
                initialW=initializers.Normal(0.01),
                initial_bias=initializers.Zero()
            )
            self.lstm = L.LSTM(hidden_size, hidden_size)
            self.out_word = L.Linear(
                hidden_size,
                vocab_size,
                initialW=initializers.Normal(0.01),
                initial_bias=initializers.Zero()
            )

        self.dropout_ratio = dropout_ratio
        self.ignore_label = ignore_label

    def step(self, word):
        h = self.embed_word(word)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        h = self.out_word(F.dropout(h, ratio=self.dropout_ratio))
        return h

    def __call__(self, img_feats, captions=None, bos=None, eos=None,
                 max_caption_length=None):
        xp = self.xp
        self.lstm.reset_state()
        h = self.embed_img(img_feats)
        self.lstm(F.dropout(h, ratio=self.dropout_ratio))

        if chainer.config.train:
            # TODO: Test Seems to work better to actually skip BOS
            # captions = captions[:, 1:]
            loss = 0
            accumulated = 0
            for i in range(captions.shape[1] - 1):
                x = Variable(xp.asarray(captions[:, i]))
                t = Variable(xp.asarray(captions[:, i + 1]))

                if (t.data == self.ignore_label).all():
                    break

                y = self.step(x)
                loss += F.sum(F.softmax_cross_entropy(
                    y, t, ignore_label=self.ignore_label, reduce='no')) / \
                    y.shape[0]
                # size += xp.sum(target)
                accumulated += 1
            return loss / max(accumulated, 1)
        return self.predict(img_feats, bos, eos, max_caption_length=10)

    def predict(self, img_feats, bos, eos, max_caption_length):
        return [self.predict_one(img_feat, bos, eos, max_caption_length)
                for img_feat in img_feats]

    def predict_one(self, img_feat, bos, eos, max_caption_length):
        caption = [bos]
        print('cap start', caption)
        while caption[-1] != eos and len(caption) < max_caption_length:
            x = Variable(self.xp.asarray([caption[-1]], dtype='i'))
            y = self.step(x)
            pred = y.data.argmax()
            caption += [int(pred)]
        print('cap', caption)
        return caption


class ImageCaptionModel(chainer.Chain):

    def __init__(self, *, vocab, img_feat_size=4096, hidden_size=512,
                 ignore_label=-1, rnn='nsteplstm'):
        super(ImageCaptionModel, self).__init__()
        vocab_size = len(vocab)
        with self.init_scope():
            self.feat_extractor = ImageFeatureExtractor(
                cnn=L.VGG16Layers(), cnn_layer_name='fc7')
            if rnn == 'lstm':
                self.lang_model = LSTMLanguageModel(
                    vocab_size, img_feat_size, hidden_size)
            elif rnn == 'nsteplstm':
                self.lang_model = NStepLSTMLanguageModel(
                    vocab_size, img_feat_size, hidden_size)
            else:
                raise ValueError()

        self.vocab = vocab
        self.finetune_cnn = False
        self.ignore_label = ignore_label

    def __call__(self, imgs, labels=None):
        imgs = Variable(imgs)
        if not self.finetune_cnn or not chainer.config.train:
            with chainer.no_backprop_mode():
                img_feat = self.feat_extractor(imgs)
        else:
            img_feat = self.feat_extractor(imgs)

        # Compute loss from language model or generate captions based on if
        # we are training or testing
        if chainer.config.train:
            loss = self.lang_model(img_feat, labels)
            reporter.report({'loss': loss}, self)
            return loss

        bos = self.vocab['<bos>']
        eos = self.vocab['<eos>']
        max_caption_length = 10
        captions = self.lang_model(img_feat, bos=bos, eos=eos,
                                   max_caption_length=max_caption_length)
        return captions
