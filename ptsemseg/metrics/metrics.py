# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import datetime
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
import scipy.ndimage


class runningScore(object):
    def __init__(self, n_classes, logger=None):
        self.logger = logger
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def print_score(self, epoch, writer=None):
        score, class_iou = self.get_scores()
        for k, v in score.items():
            if self.logger is None:
                print(k, v)
            else:
                self.logger.info("{}: {}".format(k, v))
            if writer is not None:
                writer.add_scalar("val_metrics/{}".format(k), v, epoch + 1)

        for k, v in class_iou.items():
            if self.logger is None:
                print(k, v)
            else:
                self.logger.info("{}: {}".format(k, v))
            if writer is not None:
                writer.add_scalar("val_metrics/cls_{}".format(k), v, epoch + 1)

    def __plot_confusion_matrix(self, cm, target_names, title='Pixel Confusion matrix', cmap='Oranges', fontsize=20):
        figure = plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title, fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=fontsize)

        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=fontsize)
        plt.yticks(tick_marks, target_names, fontsize=fontsize)
        plt.tight_layout(pad=5)

        width, height = cm.shape

        for x in range(width):
            for y in range(height):
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=fontsize)
        plt.gca().set_aspect('auto')
        plt.ylabel('True label', fontsize=fontsize)
        plt.xlabel('Predicted label', fontsize=fontsize)
        return figure

    def sum_to_100(self, cm, cm_org, decimals):
        num_rows, num_cols = cm.shape
        new_cm = cm.copy()
        error = 100.0 - cm.sum(axis=1)
        correction = (1 / (10 ** decimals))
        for j in range(num_rows):
            n = int(round(error[j] / correction))
            current_row = cm[j]
            org_row = cm_org[j]
            for _, i in sorted(((org_row[i] - current_row[i], i) for i in range(len(org_row))),
                               reverse=n > 0)[:abs(n)]:
                new_cm[j][i] += np.copysign(correction, n)
        if (new_cm.sum(axis=1).round(0) != 100).any():
            print("NOT SUMMING TO 100")
            print(new_cm.sum(axis=1))
            print(new_cm)

        return new_cm

    def plot_conf_matrix(self, epoch, category_names, writer=None, percentage=True, title_suffix="",
                         plot_tile_suffix="",
                         save_image = None):
        cm = self.confusion_matrix
        title = "pixel confusion matrix"
        if percentage:
            cm = cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm_org = cm
            cm = np.round(cm, 1)
            cm = self.sum_to_100(cm, cm_org, 1)
            # cm = cm.astype(np.float)
            cm = cm.round(1)
            title += " percentage"
        title += title_suffix
        fig = self.__plot_confusion_matrix(cm, category_names, title=title + plot_tile_suffix)
        img = self.__plot_to_image(fig)
        if writer is not None:
            writer.add_image(title, img, epoch, dataformats='HWC')
        if save_image is not None:
            if not Path(save_image).exists():
                Path(save_image).mkdir(parents=True)
            save_path = Path(str(save_image), str(title.replace(" ", "_")) + ".png")
            plt.imsave(str(save_path), img)

    @staticmethod
    def __plot_to_image(fig):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        # buf = io.BytesIO()
        # figure.savefig(buf, format='raw', dpi=400)
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # buf = io.BytesIO()
        # figure.savefig(buf, format='raw', dpi=400)
        # buf.seek(0)
        # img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
        #                      newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
        # buf.close()
        plt.close(fig)

        return data

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):

    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        self.st = time.time()
        self.global_st = self.st
        self.curr = self.st

    def update(self):
        self.iter += 1

    def get(self):
        self.curr = time.time()
        interv = self.curr - self.st
        global_interv = self.curr - self.global_st
        eta = int((self.max_iter - self.iter) * (global_interv / (self.iter + 1)))
        eta = str(datetime.timedelta(seconds=eta))
        self.st = self.curr
        return interv, eta
