import chainer
from chainer import functions as F
from chainer import links as L
from chainer import reporter
from chainer import cuda


class ImageCaptionModel(chainer.Chain):
    def __init__(self, vocab, img_feat_size=4096, hidden_size=512):
        super(ImageCaptionModel, self).__init__(
            feat_extractor=L.VGG16Layers(),
            lang_model=RNNLanguageModel(
                n_vocab=len(vocab),
                img_feat_size=img_feat_size,
                hidden_size=hidden_size
            )
        )
        self.vocab = vocab

    def __call__(self, imgs, labels):
        # Extract image features using VGG16
        with chainer.no_backprop_mode():
            img_feats = self.feat_extractor(imgs, ['fc7'])
            img_feat = img_feats['fc7']

        # Reset LSTM using the image features
        self.lang_model.reset_with_image_feat(img_feat)

        eos = self.vocab['<eos>']
        xp = self.xp
        loss = 0
        size = 0
        label_length = labels.shape[1]
        for i in range(label_length - 1):

            # Only consider non-eos targets
            target = xp.where(xp.asarray(labels[:, i]) != eos, 1, 0).astype(xp.float32)
            if (target == 0).all():  # eos
                break

            with chainer.using_config('train', True):
                with chainer.using_config('enable_backprop', True):
                    x = xp.asarray(labels[:, i])
                    t = xp.asarray(labels[:, i + 1])
                    y = self.lang_model(x)

                    # Ignore (no grads) for eos labels
                    mask = target.reshape((target.shape[0], 1)).repeat(y.data.shape[1], axis=1)
                    y = y * mask

                    loss += F.softmax_cross_entropy(y, t)
                    size += xp.sum(target)

        loss /= size
        reporter.report({'loss': loss}, self)

        return loss

    def predict(self, imgs, max_length=20, beam_width=20):
        xp = self.xp

        with chainer.no_backprop_mode():
            img_feats = self.feat_extractor(imgs, ['fc7'])
            img_feats = img_feats['fc7']

        bos = self.vocab['<bos>']
        eos = self.vocab['<eos>']

        n = imgs.shape[0]
        labels = xp.empty((n, beam_width, max_length), dtype=xp.int32)
        labels.fill(eos)

        for img_feat in img_feats:  # for each image
            self.lang_model.reset_with_image_feat(img_feat[xp.newaxis, :])
            candidates = [(self.lang_model.copy(), [bos], 0)]

            for _ in range(max_length):  # for each word
                next_candidates = []
                for prev_net, tokens, likelihood in candidates:  # for each candidate
                    if tokens[-1] == eos:
                        next_candidates.append((None, tokens, likelihood))
                        continue  # check next candidate
                    net = prev_net.copy()

                    # Predict next token given previous token
                    x = xp.asarray([tokens[-1]]).astype(xp.int32)
                    y = F.softmax(net(x))

                    token_likelihood = xp.log(y.data[0])
                    token_likelihood = cuda.to_cpu(token_likelihood)
                    order = token_likelihood.argsort()[:-beam_width:-1]

                    next_candidates.extend([(net, tokens + [i], likelihood + token_likelihood[i]) for i in order])

                candidates = sorted(next_candidates, key=lambda x: -x[2])[:beam_width]

                if all([candidate[1][-1] == eos for candidate in candidates]):
                    break

            # TODO: Should not return inside loop, just testing
            return candidates
        """
        labels.fill(eos)
        labels[:, 0] =
        for i in range(max_length):
            target = xp.where(xp.asarray(labels[:, i]) != eos, 1, 0).astype(xp.float32)
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    x = xp.asarray(labels[:, i])
                    t = xp.asarray(labels[:, i + 1])
                    y = self.lang_model(x)
                    y_max_idx = xp.argmax(y.data, axis=1)
                    mask = target.reshape((len(target), 1)).repeat(y.data.shape[1], axis=1)
                    y = y * mask
                    loss += F.softmax_cross_entropy(y, t)
                    size = xp.sum(target)

        loss /= size
        reporter.report({'loss': loss}, self)
        return loss
        """


class RNNLanguageModel(chainer.Chain):
    def __init__(self, n_vocab=0, img_feat_size=4096, hidden_size=512, dropout_ratio=0.5):
        super(RNNLanguageModel, self).__init__(
            embed_word=L.EmbedID(n_vocab, hidden_size),
            embed_img=L.Linear(img_feat_size, hidden_size),
            lstm=L.LSTM(hidden_size, hidden_size),  # Alt. use NStepNLSTM
            out_word=L.Linear(hidden_size, n_vocab)
        )
        self.dropout_ratio = dropout_ratio

    def reset_with_image_feat(self, img_feat):
        self.lstm.reset_state()
        h = self.embed_img(img_feat)
        h = F.relu(h)  # TODO(hvy): Really use ReLU here?
        self.lstm(F.dropout(h, ratio=self.dropout_ratio))

    def __call__(self, word):
        h = self.embed_word(word)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        h = self.out_word(F.dropout(h, ratio=self.dropout_ratio))
        return h
