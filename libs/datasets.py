import json
import os
import shutil
from abc import ABC

import numpy as np
import scipy
import scipy.io as scio
from PIL import Image, ImageFile
from randaugment import RandAugment
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    check_integrity
)
from torchvision.transforms import transforms

from libs.augmentations import Cutout

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Flowers102(VisionDataset):
    source_folder = '102flowers_org'
    base_folder = '102flowers'
    source = os.path.join(source_folder, 'jpg')
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    image_labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    set_id_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    filename = "102flowers.tgz"
    image_labels_filename = "imagelabels.mat"
    set_id_filename = "setid.mat"
    md5 = "52808999861908f626f3c1f4e79d11fa"
    image_labels_md5 = "e0620be6f572b9609742df49c70aed4d"
    set_id_md5 = "a5357ecc9cb78c4bef273ce3793fc85c"

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform=None,
            target_transform=None,
            download: bool = False,
    ):
        super(Flowers102, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        if download:
            self.download_and_arrange()

        assert (split in ('train', 'val', 'test'))
        self.split = split
        if split == 'val':
            split = 'valid'
        downloaded_list = os.path.join(self.root, self.base_folder, split)

        self.data = []
        self.targets = []

        for i in range(102):
            for file_name in os.listdir(
                    os.path.join(downloaded_list, str(i + 1))):
                if not file_name.endswith('.jpg'):
                    continue
                self.data.append(Image.open(
                    os.path.join(downloaded_list, str(i + 1), file_name)))
                self.targets.append(i)

    def __getitem__(self, index: int):

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.base_folder, self.filename)
        if not check_integrity(fpath, self.md5):
            return False
        return True

    def download_and_arrange(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(
            self.url, self.root,
            extract_root=os.path.join(self.root, self.source_folder),
            filename=self.filename, md5=self.md5)
        download_url(self.image_labels_url,
                     os.path.join(self.root, self.source_folder),
                     filename=self.image_labels_filename,
                     md5=self.image_labels_md5)
        download_url(self.set_id_url,
                     os.path.join(self.root, self.source_folder),
                     filename=self.set_id_filename, md5=self.set_id_md5)

        image_labels = scio.loadmat(os.path.join(self.root, self.source_folder,
                                                 self.image_labels_filename))
        set_id = scio.loadmat(
            os.path.join(self.root, self.source_folder, self.set_id_filename))

        self.classify(set_id['tstid'][0], 'train', image_labels['labels'][0])
        self.classify(set_id['valid'][0], 'valid', image_labels['labels'][0])
        self.classify(set_id['trnid'][0], 'test', image_labels['labels'][0])
        shutil.rmtree(os.path.join(self.root, self.source_folder))

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def classify(self, set_, split, labels):
        for n, id_ in enumerate(set_):
            cls = labels[id_ - 1]
            filename = f'image_{id_:05d}.jpg'
            dst = os.path.join(self.root, self.base_folder, split)
            path = os.path.join(dst, str(cls))
            path = path.strip()
            path = path.rstrip("/")
            os.makedirs(path, exist_ok=True)
            os.rename(os.path.join(self.root, self.source, filename),
                      os.path.join(dst, str(cls), filename))


class StanfordCars(VisionDataset):
    base_folder = 'stanford_cars'

    urls = {
        "train": "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
        "test": "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
        "devkit": "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
        "test_anno": "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
    }
    md5 = {
        "train": "065e5b463ae28d29e77c1b4b166cfe61",
        "test": "4ce7ebf6a94d07f1952d94dd34c4d501",
        "devkit": "c3b158d763b6e2245038c8ad08e45376",
        "test_anno": "b0a2b23655a3edd16d84508592a98d10",
    }

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform=None,
            target_transform=None,
            download: bool = False,
    ):
        super(StanfordCars, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.data_dir = os.path.join(self.root, self.base_folder)
        mat_anno = os.path.join(self.data_dir, 'devkit', f'cars_{split}_annos.mat') \
            if not split == "test" else os.path.join(self.data_dir,
                                                     'cars_test_annos_withlabels.mat')
        car_names = os.path.join(self.data_dir, 'devkit', 'cars_meta.mat')

        assert (split in ('train', 'test'))
        self.split = split

        if download:
            self.download()

        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])
        self.class_num = self.car_names.shape[0]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img_name = os.path.join(
            self.data_dir, f'cars_{self.split}',
            self.car_annotations[index][-1][0])

        img = Image.open(img_name).convert('RGB')
        car_class = self.car_annotations[index][-2][0][0]

        if self.transform is not None:
            img = self.transform(img)

        target = int(car_class) - 1

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.car_annotations)

    def _check_integrity(self) -> bool:
        for k in self.urls.keys():
            fpath = os.path.join(
                self.data_dir, os.path.basename(self.urls[k]))
            if not check_integrity(fpath, self.md5[k]):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        for k in self.urls.keys():
            if os.path.splitext(self.urls[k])[-1] == '.mat':
                download_url(self.urls[k], self.data_dir,
                             md5=self.md5[k])
            else:
                download_and_extract_archive(
                    self.urls[k], self.data_dir,
                    extract_root=self.data_dir,
                    md5=self.md5[k])

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class INaturalist(VisionDataset, ABC):
    base_folder = ""
    urls = {}
    md5 = {}
    year = ""

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform=None,
            target_transform=None,
            download: bool = False,
    ):
        super(INaturalist, self).__init__(root, transform=transform,
                                          target_transform=target_transform)

        self.data_dir = os.path.join(self.root, self.base_folder)
        anno = os.path.join(self.data_dir, f'{split}{self.year}.json')

        assert (split in ('train', 'val'))
        self.split = split

        if download:
            self.download()

        with open(anno) as f:
            anno = json.load(f)
            self.annotations = anno['annotations']
            self.images = anno['images']

        self.class_num = len(set([a['category_id'] for a in self.annotations]))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img_name = os.path.join(
            self.data_dir, self.images[index]['file_name'])

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        target = self.annotations[index]['category_id']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.images)

    def _check_integrity(self) -> bool:
        for k in self.urls.keys():
            fpath = os.path.join(
                self.data_dir, os.path.basename(self.urls[k]))
            if not check_integrity(fpath, self.md5[k]):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        for k in self.urls.keys():
            download_and_extract_archive(
                self.urls[k], self.data_dir,
                extract_root=self.data_dir,
                md5=self.md5[k])

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class INaturalist18(INaturalist):
    base_folder = 'i_naturalist_18'

    urls = {
        "train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz",
        "train_json": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz",
        "val_json": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz",
    }
    md5 = {
        "train": "b1c6952ce38f31868cc50ea72d066cc3",
        "train_json": "bfa29d89d629cbf04d826a720c0a68b0",
        "val_json": "f2ed8bfe3e9901cdefceb4e53cd3775d",
    }
    year = "2018"


class INaturalist19(INaturalist):
    base_folder = 'i_naturalist_19'

    urls = {
        "train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz",
        "train_json": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train2019.json.tar.gz",
        "val_json": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/val2019.json.tar.gz",
    }
    md5 = {
        "train": "c60a6e2962c9b8ccbd458d12c8582644",
        "train_json": "b06a6683537867c0d5c7a45f407a306d",
        "val_json": "5cc5509b0fe495f1c8c1362612448497",
    }
    year = "2019"


class DatasetGetter(ABC):
    def __init__(self, color_jitter, cutout_p):
        self.color_jitter = (float(color_jitter),) * 3
        self.cutout_p = cutout_p
        pass

    def get(self, path):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def image_height(self):
        raise NotImplementedError

    @property
    def image_width(self):
        raise NotImplementedError

    @property
    def channels(self):
        raise NotImplementedError


class CIFARGetter(DatasetGetter, ABC):
    def __init__(self, color_jitter, cutout_p, mean, std, size=32):
        super().__init__(color_jitter, cutout_p)
        self._size = size
        self.train_transform = transforms.Compose(
            [
                transforms.Pad(4),
                transforms.RandomCrop(32, fill=128),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(*self.color_jitter),
                RandAugment(),
                transforms.ToTensor(),
                Cutout(cutout_p),
                transforms.Normalize(mean, std),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    @property
    def image_height(self):
        return self._size

    @property
    def image_width(self):
        return self._size

    @property
    def channels(self):
        return 3


class CIFAR10Getter(CIFARGetter):
    def __init__(self, color_jitter, cutout_p, size=32):
        super().__init__(color_jitter, cutout_p,
                         mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010),
                         size=size)

    def get(self, path):
        train_ds = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=self.train_transform,
        )
        val_ds = datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=self.test_transform,
        )
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 10


class CIFAR100Getter(CIFARGetter):
    def __init__(self, color_jitter, cutout_p, size=32):
        super().__init__(color_jitter, cutout_p,
                         mean=(0.507, 0.487, 0.441),
                         std=(0.267, 0.256, 0.276),
                         size=size)

    def get(self, path):
        train_ds = datasets.CIFAR100(
            root=path,
            train=True,
            download=True,
            transform=self.train_transform,
        )
        val_ds = datasets.CIFAR100(
            root=path,
            train=False,
            download=True,
            transform=self.test_transform,
        )
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 100


class NormalGetter(DatasetGetter, ABC):
    def __init__(self, color_jitter, cutout_p, mean, std):
        super().__init__(color_jitter, cutout_p)
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.image_height, self.image_width), scale=(0.8, 1.0),
                                             ratio=(3. / 4., 4. / 3.)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(*self.color_jitter),
                RandAugment(),
                transforms.ToTensor(),
                Cutout(cutout_p),
                transforms.Normalize(mean, std),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((self.image_height, self.image_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    @property
    def image_height(self):
        return 256

    @property
    def image_width(self):
        return 224

    @property
    def channels(self):
        return 3


class Flowers102Getter(NormalGetter):
    def __init__(self, color_jitter, cutout_p):
        super().__init__(color_jitter, cutout_p,
                         mean=(0.435, 0.378, 0.287),
                         std=(0.267, 0.246, 0.270))

    def get(self, path):
        assert os.path.exists(path)
        train_ds = Flowers102(
            root=path, split="train", download=True,
            transform=self.train_transform
        )
        val_ds = Flowers102(
            root=path, split="val", download=True,
            transform=self.test_transform
        )
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 102


class StanfordCarsGetter(NormalGetter):
    def __init__(self, color_jitter, cutout_p):
        super().__init__(color_jitter, cutout_p,
                         mean=(0.467, 0.456, 0.450),
                         std=(0.295, 0.294, 0.302))

    def get(self, path):
        assert os.path.exists(path)
        train_ds = StanfordCars(
            root=path, split="train", download=True,
            transform=self.train_transform
        )
        val_ds = StanfordCars(
            root=path, split="test", download=True,
            transform=self.test_transform
        )
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 196


class INaturalist18Getter(NormalGetter):
    def __init__(self, color_jitter, cutout_p):
        super().__init__(color_jitter, cutout_p,
                         mean=(0.467, 0.456, 0.450),
                         std=(0.295, 0.294, 0.302))

    def get(self, path):
        assert os.path.exists(path)
        train_ds = INaturalist18(
            root=path, split="train", download=True,
            transform=self.train_transform
        )
        val_ds = INaturalist18(
            root=path, split="val", download=True,
            transform=self.test_transform
        )
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 8142


class INaturalist19Getter(NormalGetter):
    def __init__(self, color_jitter, cutout_p):
        super().__init__(color_jitter, cutout_p,
                         mean=(0.471, 0.482, 0.401),
                         std=(0.239, 0.233, 0.259))

    def get(self, path):
        assert os.path.exists(path)
        train_ds = INaturalist19(
            root=path, split="train", download=True,
            transform=self.train_transform
        )
        val_ds = INaturalist19(
            root=path, split="val", download=True,
            transform=self.test_transform
        )
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 1010


class ImageNetGetter(DatasetGetter):
    def __init__(self, color_jitter, cutout_p):
        super().__init__(color_jitter, cutout_p)

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(*self.color_jitter),
                RandAugment(),
                transforms.ToTensor(),
                Cutout(cutout_p),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def get(self, path):
        assert os.path.exists(path)
        train_ds = datasets.ImageNet(
            root=path, split="train", transform=self.train_transform
        )
        val_ds = datasets.ImageNet(
            root=path, split="val", transform=self.test_transform
        )
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 1000

    @property
    def image_height(self):
        return 224

    @property
    def image_width(self):
        return 224

    @property
    def channels(self):
        return 3
