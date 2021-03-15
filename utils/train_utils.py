import numpy as np
import torch
from torch import nn


class MeanAccuracy(object):
    def __init__(self, classes_num):
        super().__init__()
        self.classes_num = classes_num
        self._crt_counter = None
        self._gt_counter = None

    def reset(self):
        """
        ::功能: counter 置零
        ::输入: 无
        ::输出: 无
        """
        self._crt_counter = np.zeros((self.classes_num, self.classes_num))  # (classes_num, classes_num) float64
        self._gt_counter = np.zeros(self.classes_num)   # (classes_num,) float64

    def update(self, probs, gt_y):
        """
        ::功能: 更新 _crt_counter 和 _gt_counter
        ::输入: probs - F.softmax(logits, dim=-1).cpu().detach().numpy(), shape = (batch_size, classes_num)
                gt_y - labels.numpy() [注]labels 是 cpu() 类型, shape = (batch_size,)
        ::输出: 无
        """
        pred_y = np.argmax(probs, axis=1)   # shape = (batch_size,) 返回 probs 每行最大值的索引(即最大值所在的列), dtype = int64
        for pd_y, gt_y in zip(pred_y, gt_y):
            self._crt_counter[gt_y][pd_y] += 1  # confusion matrix
            self._gt_counter[gt_y] += 1

    def confusion(self):
        confusion = []
        for i in range(self._crt_counter.shape[0]):
            confusion.append(self._crt_counter[i] / self._gt_counter[i])
        return np.array(confusion)

    def compute(self):
        """
        ::功能: 计算 acc
        ::输入: 无
        ::输出: acc
        """
        self._gt_counter = np.maximum(self._gt_counter, np.finfo(np.float64).eps)
        # accuracy = self._crt_counter / self._gt_counter
        # mean_acc = np.mean(accuracy)
        mean_acc = self._crt_counter.diagonal().sum() / self._gt_counter.sum()
        return mean_acc


class MeanLoss(object):
    def __init__(self, batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._sum = None
        self._counter = None

    def reset(self):
        self._sum = 0
        self._counter = 0

    def update(self, loss):
        self._sum += loss * self._batch_size
        self._counter += self._batch_size

    def compute(self):
        return self._sum / self._counter


class EarlyStopping(object):
    def __init__(self, patience):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        is_best, is_terminate = True, False
        if self.best_score is None:
            self.best_score = score
        elif self.best_score >= score:
            self.counter += 1
            if self.counter >= self.patience:
                is_terminate = True
            is_best = False
        else:
            self.best_score = score
            self.counter = 0
        return is_best, is_terminate
