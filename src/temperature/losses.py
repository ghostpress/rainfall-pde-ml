import torch
from functools import reduce


def squared_error_loss(y_pred, y, reg=False, w_pred=None, coeffs=None, wdl=False, w=None):
    loss = torch.sum((y_pred-y)**2) #, dim=1)  # sum all losses together for now
    #loss = error  # torch.mean(error)

    if reg:
        assert (w_pred is not None) & (coeffs is not None)
        loss += regularize_loss(w_pred, coeffs)

    if wdl:
        assert (w_pred is not None) & (w is not None)
        w_loss = wind_loss(w_pred, w)  # torch.mean(w_error) #FIXME: is this (and the following line) correct?
        loss += w_loss

    return loss


def regularize_loss(f, coeffs):
    dt = torch.FloatTensor

    assert (len(coeffs) == 3)
    gradient = reduce(torch.add, torch.gradient(f))

    magnitude = torch.Tensor([torch.mean(torch.linalg.norm(f, axis=0, ord=2) ** 2)])
    divergence = torch.Tensor([torch.mean(gradient.sum(0) ** 2)])
    smoothness = torch.Tensor([torch.mean(torch.linalg.norm(gradient, axis=0, ord=2) ** 2)])

    reg_loss = coeffs[0]*magnitude.type(dt)[0] + coeffs[1]*divergence.type(dt)[0] + coeffs[2]*smoothness.type(dt)[0]
    return reg_loss


def wind_loss(w_pred, w):
    wind_error = torch.sum((w_pred-w)**2) #, dim=1)
    return wind_error
