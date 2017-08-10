import chainer
from chainer import functions as F
from chainer import links as L
from chainer import reporter
from chainer import cuda


class ImageCaptionModel(chainer.Chain):
    def __init__(self, *, vocab, img_feat_size=4096, hidden_size=512):
        super(ImageCaptionModel, self).__init__()
        with self.init_scope():
            self.feat_extractor = L.VGG16Layers()
            self.lang_model = RNNLanguageModel(
                n_vocab=len(vocab),
                img_feat_size=img_feat_size,
                hidden_size=hidden_size
            )
        self.vocab = vocab

    def __call__(self, imgs, labels):
        with chainer.no_backprop_mode():
            img_feat = self.feat_extractor(imgs, ['fc7'])['fc7']

        self.lang_model.reset_with_image_feat(img_feat)

        xp = self.xp

        loss, size = 0, 0
        eos = self.vocab['<eos>']
        for i in range(labels.shape[1] - 1):
            target = xp.where(xp.asarray(labels[:, i]) != eos, 1, 0) \
                .astype(xp.float32)
            if (target == 0).all():  # eos
                break

            x = xp.asarray(labels[:, i])
            t = xp.asarray(labels[:, i+1])

            y = self.lang_model(x)
            mask = target.reshape((target.shape[0], 1)).repeat(y.data.shape[1], axis=1)
            y *= mask
            loss += F.softmax_cross_entropy(y, t)
            size += xp.sum(target)

        loss /= size
        reporter.report({'loss': loss}, self)
        return loss

    def predict_one(self, img, max_length=15, beam=20):
        if img.ndim != 3:  # (c, h, w)
            raise ValueError('Predict a single image at a time')
        xp = self.xp
        img = img[xp.newaxis, :]

        bos = self.vocab['<bos>']
        eos = self.vocab['<eos>']
        results = [[]] * beam
        with chainer.using_config('train', False):
            feats = self.feat_extractor(img, ['fc7'])['fc7']
            self.lang_model.reset_with_image_feat(feats)

            results = [{
                'caption': [bos],
                'score': 0,
                'lm': self.lang_model.copy()
            }]
            for _ in range(max_length):  # for each word
                next_results = []
                for result in results:
                    caption = result['caption']
                    score = result['score']
                    lm = result['lm']

                    if caption[-1] == eos:
                        next_results.append({
                            'caption': caption,
                            'score': score,
                            'lm': None
                        })
                    else:
                        lm = lm.copy()

                        x = xp.asarray([caption[-1]]).astype('i')
                        scores = F.log_softmax(lm(x))

                        scores = scores.data[0]
                        token_likelihood = cuda.to_cpu(scores)
                        order = scores.argsort()[:-beam:-1]
                        next_results.extend([{
                                'lm': lm,
                                'caption': caption + [i],
                                'score': score + scores[i]
                            } for i in order])

                results = sorted(next_results, key=lambda x: -x['score'])[:beam]
                if all([r['caption'][-1] == eos for r in results]):
                    break

            return results

    def predict(self, imgs, max_length=20, beam=20):
        return [self.predict_one(im, max_length, beam) for im in imgs]


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
