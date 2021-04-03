import gzip
import pickle
from pathlib import Path

import requests
import torch
from torch import optim

import mnist_models
from utils import torch_nn_utils

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

dev = torch_nn_utils.return_device()


# Define operations (functions) to do with the lambda layer
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


def numpy_to_tensor(*x):
    return map(torch.tensor, x)


if __name__ == "__main__":
    batch_size = 64
    epochs = 10
    lr = 0.1  # learning rate
    """
    Load the data files
    """
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    """
    Create the DataLoader and the DataSet
    """
    x_train, y_train, x_valid, y_valid = numpy_to_tensor(x_train, y_train, x_valid, y_valid)
    train_ds = torch_nn_utils.create_dataset(x_train, y_train)
    valid_ds = torch_nn_utils.create_dataset(x_valid, y_valid)
    train_dl = torch_nn_utils.create_dataloader(train_ds, batch_size)
    valid_dl = torch_nn_utils.create_dataloader(valid_ds, batch_size * 2)
    n, c = train_ds[:][0].shape
    xb, yb = train_ds[:batch_size]
    print("Some info about the data:")
    print(train_dl)
    print(x_train, y_train)
    print(x_train.shape)
    print(train_ds[:][1].unique())
    print(xb.shape)
    print(yb.shape)

    train_dl = torch_nn_utils.WrappedDataLoader(train_dl, preprocess)
    valid_dl = torch_nn_utils.WrappedDataLoader(valid_dl, preprocess)
    # Traditional model creation
    model_logistic = mnist_models.MnistLogistic()
    # sequential model
    model_sequential = mnist_models.create_mnist_model_sequential()
    # CNN model using nn.Module
    model_cnn = mnist_models.MnistCNN()
    # Be careful to bound the optimizer to the parameters
    loss_func = torch_nn_utils.softmax_crossentrpy()
    # opt = optim.SGD(model_logistic.parameters(), lr=lr, momentum=0.9)
    # torch_nn_utils.fit(epochs, model_logistic, loss_func, opt, train_dl, valid_dl)
    opt = optim.SGD(model_cnn.parameters(), lr=lr, momentum=0.9)
    torch_nn_utils.fit(epochs, model_cnn, loss_func, opt, train_dl, valid_dl)

    exit(0)

