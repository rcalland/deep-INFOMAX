import chainer
from chainer import functions as F
from chainer import links as L


class Encoder(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(None, 64, 4)
            self.c1 = L.Convolution2D(64, 128, 4)
            self.c2 = L.Convolution2D(128, 256, 4)
            self.c3 = L.Convolution2D(256, 512, 4)
            self.linear = L.Linear(None, 64)
            #self.bn0 = L.BatchNormalization(64)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

    def __call__(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(features)))
        h = F.relu(self.bn3(self.c3(h)))
        return self.linear(h), features


class GlobalDiscriminator(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, 512)
            self.l1 = L.Linear(512, 512)
            self.l2 = L.Linear(512, 1)
            self.c0 = L.Convolution2D(None, 64, 3)
            self.c1 = L.Convolution2D(64, 32, 3)

    def __call__(self, y, M):
        h = F.relu(self.c0(M))
        h = F.reshape(self.c1(h), (y.shape[0], -1))
        h = F.concat((y, h), axis=1)

        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(None, 512, 1)
            self.c1 = L.Convolution2D(512, 512, 1)
            self.c2 = L.Convolution2D(512, 1, 1)

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, 1000)
            self.l1 = L.Linear(1000, 200)
            self.l2 = L.Linear(200, 1)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return F.sigmoid(self.l2(h))


if __name__ == "__main__":
    import numpy as np
    encoder = Encoder()
    x = np.ones((1,3,32,32), dtype=np.float32)
    y = encoder(x)
    print(y.shape)
    discriminator = Discriminator()
    d = discriminator(y)
    print(d.shape)
