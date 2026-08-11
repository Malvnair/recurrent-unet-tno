import torch
import torch.nn.functional as F

from ptsemseg.utils import handle_input_target_mismatch


def cross_entropy2d(input, target, weight=None, reduction='sum', bkargs=None):
    n, c, h, w = input.size()
    input, target = handle_input_target_mismatch(input, target)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )
    return loss


def my_cross_entropy2d(
    input,
    target,
    weight=None,
    reduction="sum",
    scale_weight=None,
    bkargs=None,
):
    return cross_entropy2d(
        input=input,
        target=target,
        weight=weight,
        reduction=reduction,
        bkargs=bkargs,
    )


def multi_step_cross_entropy2d(input, target, weight=None, reduction='sum', scale_weight=None, bkargs=None):
    if not isinstance(input, (tuple,list)):
        return cross_entropy2d(input=input, target=target, weight=weight, reduction=reduction)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    scale = 1.0 if scale_weight is None else scale_weight
    # if scale_weight == None:  # scale_weight: torch tensor type
    #     scale = 1.0
    # else:
    #     scale = scale_weight
    # print('multi_step_cross_entropy2d scale weight is {}'.format(scale))
    n_inp = len(input)
    scale_weight = torch.pow(
        scale * torch.ones(n_inp, device=input[0].device),
        torch.arange(n_inp - 1, -1, -1, device=input[0].device, dtype=input[0].dtype),
    )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, reduction=reduction
        )

    return loss


def my_multi_step_cross_entropy2d(
    input,
    target,
    weight=None,
    reduction="sum",
    scale_weight=None,
    bkargs=None,
):
    return multi_step_cross_entropy2d(
        input=input,
        target=target,
        weight=weight,
        reduction=reduction,
        scale_weight=scale_weight,
        bkargs=bkargs,
    )


def multi_scale_cross_entropy2d(
    input, target, weight=None, reduction='sum', scale_weight=None, bkargs=None
):
    if not isinstance(input, (tuple,list)):
        return cross_entropy2d(input=input, target=target, weight=weight, reduction=reduction)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    scale = 1.0 if scale_weight is None else scale_weight
    n_inp = len(input)
    scale_weight = torch.pow(
        scale * torch.ones(n_inp, device=input[0].device),
        torch.arange(n_inp - 1, -1, -1, device=input[0].device, dtype=input[0].dtype),
    )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, reduction=reduction
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  reduction='sum',
                                  bkargs=None):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   reduction=reduction):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input,
            target,
            weight=weight,
            reduction="none",
            ignore_index=250,
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            reduction=reduction,
        )
    return loss / float(batch_size)
