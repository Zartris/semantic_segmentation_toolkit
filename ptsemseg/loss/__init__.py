import functools
import logging

from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
    OhemCELoss,
    OhemCELoss2,
    CrossEntorpy2d,
    BootstrappedCrossEntropy2d,
    MultiScaleCrossEntropy2d,
    DiceLoss
)

logger = logging.getLogger("ptsemseg")

key2loss = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    "OhemCELoss": OhemCELoss2,
    "CrossEntorpy": CrossEntorpy2d,
    "BootstrappedCrossEntropy": BootstrappedCrossEntropy2d,
    "MultiScaleCrossEntropy": MultiScaleCrossEntropy2d,
    "DiceLoss": DiceLoss
}


# Deprecated, going over to module classes.
# TODO: Write it to convert to class functions.
def get_loss_function(cfg):
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d
    else:
        loss_dict = cfg["training"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)


def get_loss_class(cfg):
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return CrossEntorpy2d()
    else:
        loss_dict = cfg["training"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return key2loss[loss_name](**loss_params)
