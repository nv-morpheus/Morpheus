from collections import OrderedDict
import math
from time import time

import numpy as np

class BasicLogger(object):
    """A minimal class for logging training progress."""

    def __init__(self, fts, baseline_loss=0.0):
        """Pass a list of fts as argument."""
        self.fts = fts
        self.train_fts = OrderedDict()
        self.val_fts = OrderedDict()
        self.id_val_fts = OrderedDict()
        for ft in self.fts:
            self.train_fts[ft] = [[], []]
            self.val_fts[ft] = [[], []]
            self.id_val_fts[ft] = [[], []]
        self.n_epochs = 0
        self.baseline_loss = baseline_loss

    def training_step(self, losses):
        for i, ft in enumerate(self.fts):
            self.train_fts[ft][0].append(losses[i])

    def val_step(self, losses):
        for i, ft in enumerate(self.fts):
            self.val_fts[ft][0].append(losses[i])

    def id_val_step(self, losses):
        for i, ft in enumerate(self.fts):
            self.id_val_fts[ft][0].append(losses[i])

    def end_epoch(self):
        self.n_epochs += 1
        for i, ft in enumerate(self.fts):
            mean = np.array(self.train_fts[ft][0]).mean()
            self.train_fts[ft][1].append(mean)
            #reset train_fts log
            self.train_fts[ft][0] = []
            if len(self.val_fts[ft][0]) > 0:
                mean = np.array(self.val_fts[ft][0]).mean()
                self.val_fts[ft][1].append(mean)
                #reset val_fts log
                self.val_fts[ft][0] = []
            if len(self.id_val_fts[ft][0]) > 0:
                mean = np.array(self.id_val_fts[ft][0]).mean()
                self.id_val_fts[ft][1].append(mean)
                #reset id_val_fts log
                self.id_val_fts[ft][0] = []

class IpynbLogger(BasicLogger):
    """Plots Logging Data in jupyter notebook"""

    def __init__(self, *args, **kwargs):
        super(IpynbLogger, self).__init__(*args, **kwargs)
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        self.plt = plt
        self.clear_output = clear_output

    def end_epoch(self, val_losses=None):
        super(IpynbLogger, self).end_epoch()
        if self.n_epochs > 1:
            self.plot_progress()

    def plot_progress(self):
        self.clear_output()
        x = list(range(1, self.n_epochs+1))
        train_loss = [self.train_fts[ft][1] for ft in self.fts]
        train_loss = np.array(train_loss).mean(axis=0)
        self.plt.plot(x, train_loss, label='train loss', color='orange')

        if len(self.val_fts[self.fts[0]]) > 0:
            self.plt.axhline(
                y=self.baseline_loss,
                linestyle='dotted',
                label='baseline val loss',
                color='blue'
            )
            val_loss = [self.val_fts[ft][1] for ft in self.fts]
            val_loss = np.array(val_loss).mean(axis=0)
            self.plt.plot(x, val_loss, label='val loss', color='blue')

        if len(self.id_val_fts[self.fts[0]]) > 0:
            id_val_loss = [self.id_val_fts[ft][1] for ft in self.fts]
            id_val_loss = np.array(id_val_loss).mean(axis=0)
            self.plt.plot(x, id_val_loss, label='identity val loss', color='pink')

        self.plt.ylim(0, max(6, math.floor(2*self.baseline_loss)))
        self.plt.legend()
        self.plt.xlabel('epochs')
        self.plt.ylabel('loss')
        self.plt.show();

class TensorboardXLogger(BasicLogger):

    def __init__(self, logdir='logdir/', run=None, *args, **kwargs):
        super(TensorboardXLogger, self).__init__(*args, **kwargs)
        from tensorboardX import SummaryWriter
        import os

        if run is None:
            try:
                n_runs = len(os.listdir(logdir))
            except FileNotFoundError:
                n_runs = 0
            logdir = logdir+f'{n_runs:04d}'
        else:
            logdir = logdir + str(run)
        self.writer = SummaryWriter(logdir)
        self.n_train_step = 0
        self.n_val_step = 0
        self.n_id_val_step = 0

    def training_step(self, losses):
        self.n_train_step += 1
        losses = np.array(losses)
        for i, ft in enumerate(self.fts):
            self.writer.add_scalar('online' + f'_{ft}_' + 'train_loss', losses[i], self.n_train_step)
            self.train_fts[ft][0].append(losses[i])
        self.writer.add_scalar('online' + '_mean_' + 'train_loss', losses.mean(), self.n_train_step)

    def val_step(self, losses):
        #self.n_val_step += 1
        for i, ft in enumerate(self.fts):
            #self.writer.add_scalar(f'_{ft}_' + 'val_loss', losses[i], self.n_val_step)
            self.val_fts[ft][0].append(losses[i])

    def id_val_step(self, losses):
        #self.n_id_val_step += 1
        for i, ft in enumerate(self.fts):
            #self.writer.add_scalar(f'_{ft}_' + 'id_loss', losses[i], self.n_id_val_step)
            self.id_val_fts[ft][0].append(losses[i])

    def end_epoch(self, val_losses=None):
        super(TensorboardXLogger, self).end_epoch()

        train_loss = [self.train_fts[ft][1][-1] for ft in self.fts]
        for i, ft in enumerate(self.fts):
            self.writer.add_scalar(f'{ft}_' + 'train_loss', train_loss[i], self.n_epochs)
        train_loss = np.array(train_loss).mean()
        self.writer.add_scalar('mean_train_loss', train_loss, self.n_epochs)

        val_loss = [self.val_fts[ft][1][-1] for ft in self.fts]
        for i, ft in enumerate(self.fts):
            self.writer.add_scalar(f'{ft}_' + 'val_loss', val_loss[i], self.n_epochs)
        val_loss = np.array(val_loss).mean()
        self.writer.add_scalar('mean_val_loss', val_loss, self.n_epochs)

        id_val_loss = [self.id_val_fts[ft][1][-1] for ft in self.fts]
        for i, ft in enumerate(self.fts):
            self.writer.add_scalar(f'{ft}_' + 'train_loss', id_val_loss[i], self.n_epochs)
        id_val_loss = np.array(id_val_loss).mean()
        self.writer.add_scalar('mean_id_val_loss', id_val_loss, self.n_epochs)

    def show_embeddings(self, categories):
        for ft in categories:
            feature = categories[ft]
            cats = feature['cats'] + ['_other']
            emb = feature['embedding']
            mat = emb.weight.data.cpu().numpy()
            self.writer.add_embedding(mat, metadata=cats, tag=ft, global_step=self.n_epochs)
