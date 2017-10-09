import numpy as np

import chainer
from chainer import functions as F
from chainer import initializers
from chainer import links as L
from chainer import reporter
from chainer import Variable


class ImageCaptionModel(chainer.Chain):

    def __init__(self, vocab, img_feat_size=4096, hidden_size=512,
                 dropout_ratio=0.5, ignore_label=-1, rnn='nsteplstm'):
        super(ImageCaptionModel, self).__init__()

        vocab_size = len(vocab)
        with self.init_scope():
            self.feat_extractor = ImageFeatureExtractor(
                cnn=L.VGG16Layers(), cnn_layer_name='fc7')
            if rnn == 'lstm':
                self.lang_model = LSTMLanguageModel(
                    vocab_size, img_feat_size, hidden_size,
                    dropout_ratio=dropout_ratio,
                    ignore_label=ignore_label)
            elif rnn == 'nsteplstm':
                self.lang_model = NStepLSTMLanguageModel(
                    vocab_size, img_feat_size, hidden_size,
                    dropout_ratio=dropout_ratio)
            else:
                raise ValueError()
        self.vocab = vocab

    def __call__(self, imgs, labels=None, max_caption_length=10):
        imgs = Variable(imgs)
        with chainer.no_backprop_mode():
            img_feats = self.feat_extractor(imgs)

        # Compute loss from language model or generate captions based on if
        # we are training or testing
        if chainer.config.train:
            loss = self.lang_model(img_feats, labels)
            reporter.report({'loss': loss}, self)
            return loss

        # Test, generate a caption
        return self.lang_model(img_feats, bos=self.vocab['<bos>'],
                               eos=self.vocab['<eos>'],
                               max_caption_length=max_caption_length)


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


class LSTMLanguageModel(chainer.Chain):

    """Recurrent LSTM language model.

    Generate captions given features extracted from images.
    """

    def __init__(self, vocab_size, img_feat_size, hidden_size,
                 dropout_ratio, ignore_label):
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

    def __call__(self, img_feats, captions=None, bos=None, eos=None,
                 max_caption_length=None):
        if chainer.config.train:
            return self.train(img_feats, captions)
        return self.predict(img_feats, bos, eos, max_caption_length)

    def reset(self, img_feats):
        self.lstm.reset_state()
        h = self.embed_img(img_feats)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        return h

    def step(self, word):
        h = self.embed_word(word)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        h = self.out_word(F.dropout(h, ratio=self.dropout_ratio))
        return h

    def train(self, img_feats, captions):
        self.reset(img_feats)
        loss = 0
        size = 0
        xp = self.xp
        for i in range(captions.shape[1] - 1):
            x = Variable(xp.asarray(captions[:, i]))
            t = Variable(xp.asarray(captions[:, i + 1]))
            if (t.data == self.ignore_label).all():
                break
            y = self.step(x)
            loss += F.sum(F.softmax_cross_entropy(
                y, t, ignore_label=self.ignore_label, reduce='no')) / \
                y.shape[0]
            size += 1
        return loss / max(size, 1)

    def predict(self, img_feats, bos, eos, max_caption_length):
        self.reset(img_feats)
        captions = self.xp.empty((img_feats.shape[0], 1), dtype='i')
        captions.fill(bos)
        for i in range(100):
            x = Variable(captions[:, -1])
            y = self.step(x)
            pred = y.data.argmax(axis=1).astype('i')
            if (pred == eos).all():
                break
            captions = self.xp.hstack((captions, pred.reshape(-1, 1)))
        return captions


class NStepLSTMLanguageModel(chainer.Chain):

    """Recurrent NStepLSTM language model.

    Generate captions given features extracted from images.
    """

    def __init__(self, vocab_size, img_feat_size, hidden_size, dropout_ratio):
        super(NStepLSTMLanguageModel, self).__init__()
        with self.init_scope():
            self.embed_word = L.EmbedID(
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

    def __call__(self, img_feats, captions=None, bos=None, eos=None,
                 max_caption_length=None):
        if chainer.config.train:
            return self.train(img_feats, captions)
        return self.predict(img_feats, bos, eos, max_caption_length)

    def train(self, img_feats, captions):
        # Initial state based on image features
        h = self.embed_img(img_feats)
        h = F.split_axis(h, h.shape[0], axis=0)
        hx, cx, _ = self.lstm(None, None, h)

        caption_lens = [len(c) for c in captions]
        caption_sections = np.cumsum(caption_lens[:-1])
        xs = F.concat(captions, axis=0)
        xs = self.embed_word(xs)
        xs = F.split_axis(xs, caption_sections, axis=0)

        _, _, ys = self.lstm(hx, cx, xs)


        ys = F.concat(ys, axis=0)
        pred_captions = self.decode_caption(F.dropout(ys, self.dropout_ratio))
        ts = F.concat(captions, axis=0)
        loss = F.softmax_cross_entropy(pred_captions[:-1], ts[1:], reduce='no')
        loss = F.sum(loss) / (len(ys) - 1)
        return loss

    def predict(self, img_feats, bos, eos, max_caption_length):
        h = self.embed_img(img_feats)
        h = F.split_axis(h, h.shape[0], axis=0)
        hx, cx, _ = self.lstm(None, None, h)

        captions = self.xp.empty((img_feats.shape[0], 1), dtype='i')
        captions.fill(bos)
        for i in range(max_caption_length):
            xs = Variable(captions[:, -1])
            xs = self.embed_word(xs)
            xs = F.split_axis(xs, xs.shape[0], axis=0)
            hx, cx, ys = self.lstm(hx, cx, xs)
            ys = F.concat(ys, axis=0)
            pred = ys.data.argmax(axis=1).astype('i')
            if (pred == eos).all():
                break
            captions = self.xp.hstack((captions, pred.reshape(-1, 1)))
        return captions
