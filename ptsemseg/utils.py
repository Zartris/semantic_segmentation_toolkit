"""
Misc Utility functions
"""
import datetime
import logging
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def combine_images(images: list):
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]

    # # Fixing shape
    # channel = 1
    # for img in images:
    #     if len(img.shape) > 2:
    #         channel = 3

    result_image = images[0]
    for i in range(1, len(images)):
        result_image = cv2.hconcat([result_image, images[i]])

    return result_image


def convert_images(writer: SummaryWriter, data_loader: data, images, pred, gt, i_val, epoch, list_of_results: list,
                   name="combined", upload=True):
    bs = images.shape[0]
    for i in range(bs):
        pred_img = (data_loader.dataset.decode_segmap(pred[i]) * 255.0).astype(np.uint8)
        gt_img = (data_loader.dataset.decode_segmap(gt[i]) * 255.0).astype(np.uint8)
        img_img = (images[i] * 255.0).astype(np.uint8)
        img_img = img_img.transpose(1, 2, 0)
        # cv2.imshow("testing", img_img)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        # img_img = cv2.cvtColor(img_img, cv2.COLOR_BGR2RGB)
        # img_img = img_img[..., ::-1].copy()  # Convert to RGB
        cb_img = combine_images([img_img, pred_img, gt_img])
        cb_img_rgb = cb_img[..., ::-1].copy().astype(np.uint8)
        list_of_results.append(cb_img_rgb)
        if upload:
            writer.add_image(name + str(i_val) + "_" + str(i), cb_img_rgb, epoch, dataformats='HWC')
    return list_of_results


