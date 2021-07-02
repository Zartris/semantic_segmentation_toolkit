import torch
import torch.nn
import torch.nn.functional as F


class OhemCELoss(torch.nn.Module):

    def __init__(self, thresh, ignore_lb=250):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.ignore_lb = ignore_lb
        self.criteria = torch.nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1).cpu()
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class OhemCELoss2(torch.nn.Module):
    def __init__(self, thresh, ignore_lb=250, *args, **kwargs):
        super(OhemCELoss2, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.ignore_lb = ignore_lb
        self.criteria = torch.nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def bootstrap_xentropy_single(self, input, target, K, thresh):
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, reduction='none', ignore_index=250
        )
        sorted_loss, _ = torch.sort(loss, descending=True)

        if sorted_loss[K] > thresh:
            loss = sorted_loss[sorted_loss > thresh]
        else:
            loss = sorted_loss[:K]
        reduced_topk_loss = torch.mean(loss)

        return reduced_topk_loss

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        n_min = target[target != self.ignore_lb].numel() // 16

        return self.bootstrap_xentropy_single(input, target, n_min, self.thresh)


class CrossEntorpy2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_lb=250):
        super(CrossEntorpy2d, self).__init__()
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=weight,
                                                          size_average=size_average,
                                                          ignore_index=ignore_lb,
                                                          reduction='mean')

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        return self.self.CrossEntropyLoss(input, target)


class MultiScaleCrossEntropy2d(torch.nn.Module):
    def __init__(self, loss_th, weight=None, size_average=True, ignore_lb=250, scale_weight=[1.0, 0.4]):
        super(MultiScaleCrossEntropy2d, self).__init__()
        self.loss_th = loss_th
        self.scale_weight = scale_weight
        self.crossEntropyLoss = CrossEntorpy2d(weight, size_average, ignore_lb)

    def bootstrapped_cross_entropy2d(self, input, target, min_K):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        batch_size = input.size()[0]

        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        thresh = self.loss_th

        def _bootstrap_xentropy_single(input, target, K, thresh, weight=None, size_average=False):
            n, c, h, w = input.size()
            input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            target = target.view(-1)
            loss = F.cross_entropy(
                input, target, weight=weight, reduction='none', ignore_index=250
            )
            sorted_loss, _ = torch.sort(loss, descending=True)

            if sorted_loss[K] > thresh:
                loss = sorted_loss[sorted_loss > thresh]
            else:
                loss = sorted_loss[:K]
            reduced_topk_loss = torch.mean(loss)

            return reduced_topk_loss

        loss = 0.0
        # Bootstrap from each image not entire batch
        for i in range(batch_size):
            loss += _bootstrap_xentropy_single(
                input=torch.unsqueeze(input[i], 0),
                target=torch.unsqueeze(target[i], 0),
                K=min_K,
                thresh=thresh,
                weight=self.weight,
                size_average=self.size_average,
            )
        return loss / float(batch_size)

    def forward(self, input, target):
        if not isinstance(input, tuple):
            return self.crossEntropyLoss(input, target)

        K = input[0].size()[2] * input[0].size()[3] // 128
        loss = 0.0

        for i, inp in enumerate(input):
            loss = loss + self.scale_weight[i] * self.bootstrapped_cross_entropy2d(
                input=inp, target=target, min_K=K)

        return loss


class BootstrappedCrossEntropy2d(torch.nn.Module):
    def __init__(self, end_k_percentage, loss_th, weight, start_warm=2000, end_warm=5000):
        """
        Quote:"https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch"
            Often we would also add a "warm-up" period to the loss such that the network
            can learn to adapt to the easy regions first and transit to the harder regions.
            This implementation starts from k=100 and continues for 20000 iterations,
            then linearly decay it to k=15 for another 50000 iterations.
        @param end_k_percentage:
        @param loss_th:
        @param weight: Not supported.
        @param start_warm:
        @param end_warm:
        """
        super(BootstrappedCrossEntropy2d, self).__init__()
        self.end_k_percentage = end_k_percentage
        self.loss_th = loss_th
        self.weight = torch.tensor(weight if weight is not None else [1.0]).to('cuda')
        self.start_warm = start_warm
        self.end_warm = end_warm
        self.epoch = 0

    def bootstrap_xentropy_single(self, input, target, K, thresh, weight=None):
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduction='none', ignore_index=250
            # Changed from none to mean
        )
        if weight is not None:  # Maybe this undo's the weights, not sure.
            loss = loss.sum() / weight[target].sum()
        sorted_loss, _ = torch.sort(loss, descending=True)

        if sorted_loss[K] > thresh:
            loss = sorted_loss[sorted_loss > thresh]
        else:
            loss = sorted_loss[:K]
        reduced_topk_loss = torch.mean(loss)

        return reduced_topk_loss

    def forward(self, input, target):
        self.epoch += 1
        if len(self.weight) == 1:
            input = [input]
        assert len(self.weight) == len(input)
        n, c, h, w = input[0].size()
        nt, ht, wt = target.size()
        batch_size = n

        if h != ht and w != wt:  # upsample labels
            input = [F.interpolate(i, size=(ht, wt), mode="bilinear", align_corners=True) for i in input]

        thresh = self.loss_th
        loss = 0.0
        # Calculated the new K
        number_of_pixels = ht * wt
        K = number_of_pixels - 1
        if self.start_warm < self.epoch <= self.end_warm:
            this_percentage = self.end_k_percentage + (1 - self.end_k_percentage) * (
                    (self.end_warm - self.epoch) / (self.end_warm - self.start_warm))
            K = int(number_of_pixels * this_percentage)
        elif self.end_warm < self.epoch:
            K = int(number_of_pixels * self.end_k_percentage)

        # Bootstrap from each image not entire batch
        for i in range(batch_size):
            for j in range(len(input)):
                loss += self.weight[j] * self.bootstrap_xentropy_single(
                    input=torch.unsqueeze(input[j][i], 0),
                    target=torch.unsqueeze(target[i], 0),
                    K=K,
                    thresh=thresh,
                    weight=None
                )
        return loss / float(batch_size)


# Depricated
class BootstrappedCE(torch.nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class DiceLoss(torch.nn.Module):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """

    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.ignore_index = 250

    def dice_loss(self, input, target):
        num_classes = input.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(input)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            # TODO: Implement ignore index
            # mask = target == self.ignore_index
            true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(input, dim=1)
        true_1_hot = true_1_hot.type(input.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        # input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        # target = target.view(-1)
        return self.dice_loss(input, target)


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250, reduction='mean')

    return loss


def multi_scale_cross_entropy2d(input, target, loss_th, weight=None, size_average=True, scale_weight=[1.0, 0.4]):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    K = input[0].size()[2] * input[0].size()[3] // 128
    loss = 0.0

    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * bootstrapped_cross_entropy2d(
            input=inp, target=target, min_K=K, loss_th=loss_th, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, min_K, loss_th, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    batch_size = input.size()[0]

    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    thresh = loss_th

    def _bootstrap_xentropy_single(input, target, K, thresh, weight=None, size_average=False):
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=size_average, ignore_index=250
        )
        sorted_loss, _ = torch.sort(loss, descending=True)

        if sorted_loss[K] > thresh:
            loss = sorted_loss[sorted_loss > thresh]
        else:
            loss = sorted_loss[:K]
        reduced_topk_loss = torch.mean(loss)

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=min_K,
            thresh=thresh,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)


def bce_loss(true, logits, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, 1, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        true.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


def ce_dice(true, pred, log=False, w1=1, w2=1):
    pass


def ce_jaccard(true, pred, log=False, w1=1, w2=1):
    pass


def focal_loss(true, pred):
    pass
