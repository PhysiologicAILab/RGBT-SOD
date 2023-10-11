import numbers
import random

from PIL import Image, ImageOps
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, thermal,mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, thermal,mask = t(img, thermal,mask)
        return img, thermal,mask


class RandomCrop(object):
    def __init__(self, size,size1, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size1))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, thermal,mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            thermal = ImageOps.expand(thermal, border=self.padding, fill=0)
            
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img,thermal, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR),thermal.resize((tw, th), Image.NEAREST), mask.resize((tw, th), Image.NEAREST)
        return img.resize((tw, th), Image.BILINEAR), thermal.resize((tw, th), Image.NEAREST),mask.resize((tw, th), Image.NEAREST)


class RandomHorizontallyFlip(object):
    def __call__(self, img, thermal,mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), thermal.transpose(Image.FLIP_LEFT_RIGHT),mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, thermal,mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, thermal,mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR),thermal.rotate(rotate_degree, Image.NEAREST), mask.rotate(rotate_degree, Image.NEAREST)
