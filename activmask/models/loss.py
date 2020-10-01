import torch
import numpy as np
import torch.nn.functional as F


def compare_activations(act_a, act_b):
    """
    Calculates the mean l2 norm between the lists of activations
    act_a and act_b.
    """
    assert len(act_a) == len(act_b)
    dist = torch.nn.modules.distance.PairwiseDistance(p=2)
    all_dists = []

    # Gather all L2 distances between the activations
    for a, b in zip(act_a, act_b):
        all_dists.append(dist(a, b).view(-1))

    all_dists = torch.cat(all_dists)
    actdiff_loss = all_dists.sum() / len(all_dists)

    return(actdiff_loss)


def get_grad_contrast(X, y_pred):
    """Gradmask: Simple Constrast Loss. d(y_0-y_1)/dx"""
    contrast = torch.abs(y_pred[:, 0] - y_pred[:, 1])
    # This is always a list of length 1, so we remove the element from the list.
    gradients = torch.autograd.grad(
        outputs=contrast, inputs=X, allow_unused=True, create_graph=True,
        grad_outputs=torch.ones_like(contrast))[0]

    return gradients


def get_grad_rrr(X, y_pred):
    """Right for the Right Reasons."""
    EPS = 10e-12
    y_pred = torch.sum(torch.log(F.softmax(y_pred, dim=1) + EPS))
    # This is always a list of length 1, so we remove the element from the list.
    gradients = torch.autograd.grad(
        outputs=y_pred, inputs=X, allow_unused=True, create_graph=True)[0]

    return gradients


def get_grad_saliency(X, y_pred):
    """Simple saliency map for the positive class. d(y_1)/dx."""
    y_pred = y_pred[:, 1].sum()
    # This is always a list of length 1, so we remove the element from the list.
    gradients = torch.autograd.grad(
        outputs=y_pred, inputs=X, allow_unused=True, create_graph=True)[0]

    return gradients
