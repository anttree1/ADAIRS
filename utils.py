import torch


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target):
    maxk = max((1,))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].contiguous().view(-1).float().sum(0)
    correct_k.mul_(100.0 / batch_size)
    return correct_k


def sensitivity(output, target):
    out = torch.argmax(output, dim=1).float()

    equal1 = torch.logical_and(out == target, target == 1.)
    tp = torch.sum(equal1).item()

    equal2 = torch.logical_and(out != target, target == 1.)
    fn = torch.sum(equal2).item()

    equal3 = torch.logical_and(out == target, target == 0.)
    tn = torch.sum(equal3).item()

    equal4 = torch.logical_and(out != target, target == 0.)
    fp = torch.sum(equal4).item()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    return sensitivity, specificity

