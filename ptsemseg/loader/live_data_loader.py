import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data

from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from ptsemseg.utils import recursive_glob


class harbourLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    categories = {
        "unknown": (128, 219, 130),
        "water": (173, 141, 224),
        "ship": (114, 119, 232)
    }
    ignore_categories = {
    }

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
            self,
            root,
            split="train",
            is_transform=False,
            img_size=(1024, 2048),
            augmentations=None,
            img_norm=True,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = len(self.categories.keys())
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}

        self.images_base = Path(self.root, "model_base", "images")
        self.annotations_base = Path(self.root, "model_base", "label")

        if not self.images_base.exists():
            Path(self.images_base, "train").mkdir(parents=True)
            Path(self.images_base, "val").mkdir(parents=True)
            Path(self.images_base, "test").mkdir(parents=True)

        if not self.annotations_base.exists():
            Path(self.annotations_base, "train").mkdir(parents=True)
            Path(self.annotations_base, "val").mkdir(parents=True)
            Path(self.annotations_base, "test").mkdir(parents=True)


        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.ignore_index = 250  # When rotating, this color is padded
        self.class_map = dict(zip(self.categories.keys(), range(self.n_classes)))
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        self.generate_config()
        print("Found %d %s images" % (len(self.files[split]), split))

    def val_dataset(self):
        self.split = "val"

    def train_dataset(self):
        self.split = "train"

    def test_dataset(self):
        self.split = "test"


    def update_files(self):
        self.files[self.split] = recursive_glob(rootdir=self.images_base, suffix=".png")

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = Path(self.files[self.split][index].rstrip())
        name = img_path.stem + "_color_mask" + img_path.suffix
        lbl_path = Path(self.annotations_base, name)

        img = Image.open(str(img_path))
        img = np.array(img, dtype=np.uint8)
        if img.shape[-1] == 4:  # check if RGBA
            img = img[:, :, :3]  # Remove alpha
        img = img.astype(np.uint8)

        lbl = Image.open(str(lbl_path))
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        lbl = lbl.astype(np.uint8)
        # cv2.imshow("asd", img)
        # cv2.imshow("asdasd", lbl)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        # cv2.imshow("asd", img)
        # cv2.imshow("asdasd", lbl)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, name

    def transform(self, img, lbl):
        """ transform

        :param img:
        :param lbl:
        """
        img = np.array(Image.fromarray(img).resize(
            (self.img_size[1], self.img_size[0])))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        # cv2.imshow("blah", img)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        img = img / 255
        img = img.astype(np.float64)

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = np.array(Image.fromarray(lbl).resize(
            (self.img_size[1], self.img_size[0]), resample=Image.NEAREST))
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        """
        Transform from id to colors
        """
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for category in self.categories:
            class_index = self.class_map[category]
            r[temp == class_index] = self.categories[category][0]
            g[temp == class_index] = self.categories[category][1]
            b[temp == class_index] = self.categories[category][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def decode_segmap_id(self, temp):
        """
        Remove ignored classes
        """
        ids = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
        for category in self.categories:
            if category in self.ignore_categories:
                continue
            class_index = self.class_map[category]
            ids[temp == class_index] = class_index
        return ids

    def encode_segmap(self, mask):
        """
        Transform from colors to id.
        """
        new_mask = np.zeros(mask.shape[:2])
        # Put all void classes to zero
        for category in self.categories:
            if category in self.ignore_categories:
                continue
            color = self.categories[category]
            class_index = self.class_map[category]
            color_index = np.where(
                (mask[:, :, 0] == color[0]) & (mask[:, :, 1] == color[1]) & (mask[:, :, 2] == color[2]))
            new_mask[color_index] = class_index
        return new_mask

    def generate_config(self):
        config_file = Path(self.root, "config.json")
        if config_file.exists():
            return
        config = {}
        for category in self.categories:
            if category in self.ignore_categories:
                continue
            color = self.categories[category]
            class_index = self.class_map[category]
            config[category] = {"categorie": category,
                                "color": [
                                    color[0],
                                    color[1],
                                    color[2]
                                ],
                                "id": class_index,
                                "id_categorie": class_index,
                                "name": category
                                }
        with open(str(config_file), 'w') as fp:
            json.dump(config, fp)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(640), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/home/zartris/Downloads/rosbag/img_labels/train/images_sorted/"
    dst = harbourLoader(local_path, img_size=(720, 1280), is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels, names = data_samples

        # pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
