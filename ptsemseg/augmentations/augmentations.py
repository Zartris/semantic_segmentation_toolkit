import math
import numbers
import random

import numpy as np
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask


class SaltAndPepperNoise(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, mask):
        """salt and pepper noise"""
        img = np.array(img)
        shape = img.shape[:2]
        rnd = np.random.rand(shape[0], shape[1])
        noisy = img[:]
        noisy[rnd < self.prob / 2] = 0
        noisy[rnd > 1 - self.prob / 2] = 255
        # cv2.imshow("blah", noisy)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        noisy = Image.fromarray(noisy, 'RGB')
        return noisy, mask


class GaussianNoise(object):
    def __init__(self, args):
        self.mean = args[0]
        self.sigma = args[1]
        print("sigma", self.sigma)

    def __call__(self, img, mask):
        img = np.array(img)
        shape = img.shape
        gaussian = np.random.normal(self.mean, self.sigma, shape)  # np.zeros((224, 224), np.float32)
        noisy_image = np.zeros(img.shape, np.float32)

        if len(img.shape) == 2:
            noisy_image = img + gaussian
        else:
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian[:, :, 0]
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian[:, :, 1]
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian[:, :, 2]
        noisy_image = noisy_image.clip(0, 255)
        noisy_image = noisy_image.astype(np.uint8)
        # cv2.imshow("asdasdasd", noisy_image)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        noisy_image = Image.fromarray(noisy_image, 'RGB')

        return noisy_image, mask


class SwapChannels(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, mask):
        """swap RGB to BGR"""
        rnd = random.random()
        np_img = np.array(img)
        if len(np_img.shape) != 2 and rnd <= self.prob:
            # scramble = [0,1,2]
            # random.shuffle(scramble)
            # np_img = np_img[:, :, scramble] # scramble all channels
            np_img = np_img[:, :, ::-1]  # swap R and B channel
            img = Image.fromarray(np_img, 'RGB')
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        ch, cw = self.size
        if w == cw and h == ch:
            return img, mask
        if w < cw or h < ch:
            pw = cw - w if cw > w else 0
            ph = ch - h if ch > h else 0
            padding = (int(np.floor(pw / 2)), int(np.floor(ph / 2)), int(np.ceil(pw / 2)), int(np.ceil(ph / 2)))
            img = ImageOps.expand(img, padding, fill=0)
            mask = ImageOps.expand(mask, padding, fill=250)
            w, h = img.size
            assert img.size == mask.size

        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        return (img.crop((x1, y1, x1 + cw, y1 + ch)), mask.crop((x1, y1, x1 + cw, y1 + ch)))


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)),
            mask,
        )


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT))
        return img, mask


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM))
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST))


class RandomScaleCrop(object):
    def __init__(self, size, scale_min=0.5, scale_max=2.0):
        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        r = random.uniform(self.scale_min, self.scale_max)
        w, h = img.size
        new_size = (int(w * r), int(h * r))
        return self.crop(img.resize(new_size, Image.BILINEAR), mask.resize(new_size, Image.NEAREST))


class RandomTranslate(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(
            img,
            y_crop_offset,
            x_crop_offset,
            img.size[1] - abs(y_offset),
            img.size[0] - abs(x_offset),
        )

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
            tf.pad(cropped_img, padding_tuple, padding_mode="reflect"),
            tf.affine(
                mask,
                translate=(-x_offset, -y_offset),
                scale=1.0,
                angle=0.0,
                shear=0.0,
                fillcolor=250,
            ),
        )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=(0, 0, 0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=250,
                shear=0.0,
            ),
        )


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST))

        return self.crop(*self.scale(img, mask))


class ColorJitter(object):
    def __init__(self, args):
        self.brightness = args['brightness'] if 'brightness' in args else None
        self.contrast = args['contrast'] if 'contrast' in args else None
        self.saturation = args['saturation'] if 'saturation' in args else None
        self.color_drop = args['color_drop'] if 'color_drop' in args else None
        if not self.brightness is None and self.brightness >= 0:
            self.brightness = [max(1 - self.brightness, 0), 1 + self.brightness]
        if not self.contrast is None and self.contrast >= 0:
            self.contrast = [max(1 - self.contrast, 0), 1 + self.contrast]
        if not self.saturation is None and self.saturation >= 0:
            self.saturation = [max(1 - self.saturation, 0), 1 + self.saturation]
        if not self.color_drop is None and self.color_drop >= 0:
            self.color_drop = [max(1 - self.color_drop, 0), 1 + self.color_drop]
        self.RGB2Grayscale = torchvision.transforms.Grayscale(num_output_channels=3)

    def __call__(self, img, mask):
        assert img.size == mask.size
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            img = self.adj_brightness(img, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            img = self.adj_contrast(img, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            img = self.adj_saturation(img, rate)
        if not self.color_drop is None:
            if np.random.uniform(0, 1) <= self.color_drop:
                img = self.RGB2Grayscale(img)
        return img, mask

    def adj_saturation(self, im: Image, rate):
        M = np.float32([
            [1 + 2 * rate, 1 - rate, 1 - rate],
            [1 - rate, 1 + 2 * rate, 1 - rate],
            [1 - rate, 1 - rate, 1 + 2 * rate]
        ])
        shape = im.size
        im = np.matmul(im.reshape(-1, 3), M).reshape(shape) / 3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def adj_brightness(self, im, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]
