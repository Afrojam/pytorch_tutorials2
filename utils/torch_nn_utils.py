import gzip
import pickle

import numpy as np
import requests
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from torch.utils.data import TensorDataset, DataLoader, Dataset

print(torch.cuda.is_available())

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def return_device():
    return dev


def create_dataset(x, y):
    # Permits to return x and y from a tensor as a tuple.
    return TensorDataset(x, y)


def create_dataloader(dataset, batch_size=64, num_workers=0, shuffle_data=False):
    # Manages the batches
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data)


def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs * 2))


"""
Dataset classes
"""


class DatasetMNIST(Dataset):

    def __init__(self, url, file_path, file, partition_type, transform=None):
        if not (file_path / file).exists():
            content = requests.get(url + file).content
            (file_path / file).open("wb").write(content)

        if partition_type == "train":
            with gzip.open((file_path / file).as_posix(), "rb") as f:
                ((self.x, self.y), _, _) = pickle.load(f, encoding="latin-1")
        if partition_type == "validation":
            with gzip.open((file_path / file).as_posix(), "rb") as f:
                (_, (self.x, self.y), _) = pickle.load(f, encoding="latin-1")
        if partition_type == "test":
            with gzip.open((file_path / file).as_posix(), "rb") as f:
                (_, _, (self.x, self.y)) = pickle.load(f, encoding="latin-1")
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.x[index].reshape((1, 28, 28))
        label = self.y[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def show_image(self, index):
        image = self.x[index].reshape((1, 28, 28))
        pyplot.imshow(image[index], cmap="gray")


"""
Classes Transform  to use with the DataSet class
"""


class ToTensor():
    def __call__(self, *array):
        return map(torch.tensor, array)


"""
Metrics functions
"""


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)  # Take the index of the greatest value
    return (preds == yb).float().mean()


"""
Loss functions
"""


def softmax_crossentrpy():
    return F.cross_entropy  # Combines the softmax with the cross entropy loss.


"""
Train functions
"""


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        valid_loss = evaluate(model, loss_func, valid_dl)

        print(f"Validation loss: {valid_loss} in Epoch {epoch+1}")


def evaluate(model, loss_func, test_dl):
    model.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl]
        )
    return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        a = list(model.parameters())[0].clone()
        opt.step()
        opt.zero_grad()
        b = list(model.parameters())[0].clone()
        torch.equal(a,b)
    return loss.item(), len(xb)


"""
Auxiliary classes
"""


class WrappedDataLoader:
    """
    Generator that given a Dataloader object performs an operation defined in func when data is loaded.
    """

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)
