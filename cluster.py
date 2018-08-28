import os
import argparse
import chainer

from chainer import serializers, iterators
from chainer import functions as F
from chainer.dataset import concat_examples
from networks import Encoder, LocalDiscriminator, GlobalDiscriminator

import numpy as np
import matplotlib.pyplot as plt


def main(args):
    train, test = chainer.datasets.get_cifar10()
    test_iter = iterators.SerialIterator(test, 1, shuffle=False, repeat=False)
    train_iter = iterators.SerialIterator(train, 1, shuffle=False, repeat=False)

    encoder = Encoder()
    serializers.load_npz(args.input, encoder)

    if args.device >= 0:
        encoder.to_gpu(args.device)
    else:
        raise ValueError("Currently only GPU mode works, sorry!")

    _t = -1
    while _t != args.label:
        test_batch = test_iter.next()
        x, t = concat_examples(test_batch, args.device)
        key, f = encoder(x)
        _t = t.get().tolist()[0]

    distance = []
    features = []
    truth = []
    image = []

    c = 0
    with chainer.using_config('train', False):
        #for i in range(500):
        #train_batch = test_iter.next()
        for train_batch in test_iter:
            _x, _t = concat_examples(train_batch, args.device)
            _y, _f = encoder(_x)

            dist = F.mean_absolute_error(key, _y).data.get().flatten().tolist()[0]
            true = _t.get().tolist()[0]
            pic = _x.get()[0].transpose(1, 2, 0)
            #print(dist, true, pic.shape)

            distance.append(dist)
            truth.append(true)
            image.append(pic)
            c += 1

            if c % 1000 == 0:
                print(c)

    idx = sorted(range(len(distance)),key=distance.__getitem__)

    for i in idx[:10]:
        print(distance[i], truth[i])

    print("original", t)

    img = x.get()[0].transpose(1, 2, 0)

    middle_row = np.concatenate([image[i] for i in idx[:11]], axis=1)
    top_row = np.concatenate(((img,) +  tuple([np.ones_like(img) for i in range(10)])), axis=1)
    bottom_row = np.concatenate(tuple([image[i] for i in idx[-11:]]), axis=1)

    _img = np.concatenate((top_row, middle_row, bottom_row), axis=0)
    plt.imshow(_img)

    plt.axis('off')
    plt.show()
    plt.imsave(os.path.join(args.output, "img.png"), _img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Specify the encoder model file")
    parser.add_argument("--output", "-o", type=str, default=".", help="Specify the folder where output images are saved")
    parser.add_argument("--label", "-l", type=int, default=1, help="Specify the class label to cluster")
    parser.add_argument("--device", "-g", type=int, default=-1)
    args = parser.parse_args()
    main(args)
