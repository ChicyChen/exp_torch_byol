import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms as T

class ImageNet32():
    base_folder = "ImageNet32"
    train_list = [
        "train_data_batch_1",
        "train_data_batch_2",
        "train_data_batch_3",
        "train_data_batch_4",
        "train_data_batch_5",
        "train_data_batch_6",
        "train_data_batch_7",
        "train_data_batch_8",
        "train_data_batch_9",
        "train_data_batch_10"
    ]
    test_list = [
        "val_data"
    ]
    # meta = {
    #     "filename": "meta.bin",
    #     "key": "label_names"
    # }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__()

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data: Any = []
        self.targets = []

        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
                
        
        self.data = np.vstack(self.data)
        # print(self.data.shape)
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # self._load_meta()

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta["filename"])
    #     with open(path, "rb") as infile:
    #         data = pickle.load(infile, encoding="latin1")
    #         self.classes = data[self.meta["key"]]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class ImageNet64():
    base_folder = "ImageNet64"
    train_list = [
        "train_data_batch_1",
        "train_data_batch_2",
        "train_data_batch_3",
        "train_data_batch_4",
        "train_data_batch_5",
        "train_data_batch_6",
        "train_data_batch_7",
        "train_data_batch_8",
        "train_data_batch_9",
        "train_data_batch_10"
    ]
    test_list = [
        "val_data"
    ]
    # meta = {
    #     "filename": "meta.bin",
    #     "key": "label_names"
    # }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__()

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data: Any = []
        self.targets = []

        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        
        self.data = np.vstack(self.data)
        # print(self.data.shape)
        self.data = self.data.reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # self._load_meta()

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta["filename"])
    #     with open(path, "rb") as infile:
    #         data = pickle.load(infile, encoding="latin1")
    #         self.classes = data[self.meta["key"]]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def test():
    transform = T.Compose([
            T.ToTensor(),          
        ])
    dataset = ImageNet32(root="./", train=False, transform=transform)
    # dataset = ImageNet64(root="./", train=False, transform=transform)
    print(len(dataset))
    # val_data = DataLoader(dataset=dataset, batch_size=8, num_workers=4, drop_last=True)
    # i = 0
    # for data in val_data:
    #     images, label = data
    #     # print(images, label)
    #     transform_back = T.ToPILImage()
    #     images0 = transform_back(images[0])
    #     images0.save("vis%s.jpg" % i)
    #     i += 1
    #     if i >= 20:
    #         break


if __name__ == '__main__':
    test()
