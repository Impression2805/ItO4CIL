import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                         decimals=2)

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


class calibratedCE(nn.Module):
    def __init__(self, cls_num_list, num_old, num_all):
        super(calibratedCE, self).__init__()
        beta = num_old * [cls_num_list[0] / sum(cls_num_list)] + (num_all - num_old) * [1]
        alpha = [cls_num_list[-1] / cls_num for cls_num in cls_num_list]
        alpha = np.array(alpha) ** 0.3
        beta_classwise = np.log(beta)
        alpha_classwise = alpha

        self.beta_classwise = torch.cuda.FloatTensor(beta_classwise)
        self.alpha_classwise = torch.cuda.FloatTensor(alpha_classwise)

    def forward(self, x, target, device):
        # output = x / self.alpha_classwise.to(device)
        output = x + self.beta_classwise.to(device)
        return F.cross_entropy(output, target)



class ForgetAwareAWP(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ForgetAwareAWP, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]),p=2)
        return norm
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups