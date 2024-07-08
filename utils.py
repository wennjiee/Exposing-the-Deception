from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from metrics.auc_metric import AUCMetric
from metrics.logloss_metric import LOGLOSSMetric
import torch
from torch.utils.tensorboard import SummaryWriter
import logging

class plt_tensorboard():
    def __init__(self, args):
        if not args.test:
            wr = SummaryWriter(log_dir=f'runs/train_{args.name}')
            self.tb_writer = wr
        self.metrics = [AccuracyMetric(), TestLossMetric(), AUCMetric(), LOGLOSSMetric()]


    def accumulate_metrics(self, outputs, labels, loss):
        self.metrics[0].accumulate_on_batch([outputs, labels]) # ACC
        self.metrics[1].accumulate_on_batch(loss)              # TestLoss
        self.metrics[2].accumulate_on_batch([outputs, labels]) # AUC
        self.metrics[3].accumulate_on_batch([outputs, labels]) # LogLoss

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()
    
    def report_for_cifar(self, step, tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        metric_text.append(str(self.metrics[0]))
        self.metrics[0].plot(tb_writer, step, tb_prefix=tb_prefix)
        print(f"ACC: {self.metrics[0].get_value()['value']}")
        return None, self.metrics[0].get_value()['value']

    def report_metrics(self, step, tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        for metric in self.metrics:
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)
        try:
            auc = self.metrics[2].get_value()['value']
            acc = self.metrics[0].get_value()['value']
            logloss = self.metrics[3].get_value()['value']
            print(f"AUC: {auc}, "
                  f"ACC: {acc}, "
                  f"LogLoss: {logloss}")
        except:
            print("error")
        return auc, acc
    
    def report_test_metrics(self):
        auc = self.metrics[2].get_value()['value']
        acc = self.metrics[0].get_value()['value']
        logloss = self.metrics[3].get_value()['value']
        try:
            print(f"AUC: {auc}, "
                  f"ACC: {acc}, "
                  f"LogLoss: {logloss}")
        except:
            print("error")
        return auc, acc
    
    def write_tensorboard(self, step, tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        for metric in self.metrics:
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def create_table(params):
    data = "starting logging | name | value | \n |-----|-----|"
    params=params.__dict__
    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data