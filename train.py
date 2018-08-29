import argparse
import chainer
from chainer import iterators, optimizers, serializers, reporter, training
from chainer.training import extensions
from chainer import functions as F
from chainer.dataset import concat_examples

from networks import Encoder, LocalDiscriminator, GlobalDiscriminator, PriorDiscriminator


class DeepINFOMAX(chainer.Chain):

    def __init__(self, alpha=1., beta=1., gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        with self.init_scope():
            self.encoder = Encoder()
            self.local_disc = LocalDiscriminator()
            self.global_disc = GlobalDiscriminator()
            self.prior_disc = PriorDiscriminator()

    def __call__(self, x, t):
        # get encodings
        y, M = self.encoder(x)

        # shuffle batch to pair each element with another
        M_prime = F.concat((M[1:], (M[0])[None,:,:,:]), axis=0)

        # local DIM
        y_M = F.concat((F.broadcast_to(y[:, :, None, None], \
                                       (x.shape[0], y.shape[1], M.shape[-2], M.shape[-1])), M), axis=1)
        y_M_prime = F.concat((F.broadcast_to(y[:, :, None, None], \
                                             (x.shape[0], y.shape[1], M.shape[-2], M.shape[-1])), M_prime), axis=1)

        Ej = F.mean(-F.softplus(-self.local_disc(y_M)))
        Em = F.mean(F.softplus(self.local_disc(y_M_prime)))
        local_loss = (Em - Ej) * self.beta

        # global DIM
        Ej = F.mean(-F.softplus(-self.global_disc(y, M)))
        Em = F.mean(F.softplus(self.global_disc(y, M_prime)))
        global_loss = (Em - Ej) * self.alpha

        # prior term
        z = self.xp.random.uniform(size=y.shape).astype(self.xp.float32)
        
        term_a = F.mean(F.log(self.prior_disc(z)))
        term_b = F.mean(F.log(1. - self.prior_disc(y)))
        prior_loss = -(term_a + term_b) * self.gamma

        loss = global_loss + local_loss + prior_loss

        reporter.report({"loss": loss, "local_loss": local_loss, "global_loss": global_loss, "prior_loss": prior_loss}, self)
        return loss


def main(args):
    train, test = chainer.datasets.get_cifar10()
    train_iter = iterators.SerialIterator(train, args.batchsize)

    dim = DeepINFOMAX(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    if args.device >= 0:
        chainer.backends.cuda.get_device_from_id(args.device).use()
        dim.to_gpu(args.device)

    opt = optimizers.Adam(alpha=args.learning_rate)
    opt.setup(dim)

    updater = training.updaters.StandardUpdater(
        train_iter, opt, device=args.device)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.output)

    log_interval = (10, "iteration")
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/local_loss', 'main/global_loss', 'main/prior_loss', 'elapsed_time']), trigger=log_interval)

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=log_interval[0]))

    trainer.extend(extensions.snapshot_object(dim.encoder, 'encoder_epoch_{.updater.epoch}'), trigger=(100, "epoch"))

    # Run the training
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-g", type=int, default=-1)
    parser.add_argument("--epochs", "-e", type=int, default=1000)
    parser.add_argument("--batchsize", "-b", type=int, default=256)
    parser.add_argument("--learning_rate", "-l", type=float, default=1.E-4)
    parser.add_argument("--output", "-o", type=str, default="results")
    parser.add_argument("--alpha", "-A", type=float, default=0.5)
    parser.add_argument("--beta", "-B", type=float, default=1.0)
    parser.add_argument("--gamma", "-G", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
