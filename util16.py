import numpy as np
import cupy as cp

import torch
import torchvision


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


def load_torchMnist():
    print("[INFO] loading torch MNIST (sample) dataset...")
    train_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
    )
    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
    )

    tr_x = train_data.data / 255
    tr_y = train_data.targets

    ts_x = test_data.data / 255
    ts_y = test_data.targets

    tr_x = cp.asarray(tr_x, dtype=cp.float16).reshape(60000, -1)
    print(f"[INFO]\t(cupy)tr_x:  {tr_x.shape}")

    tr_y = cp.asarray(tr_y, dtype=cp.int8)
    print(f"[INFO]\t(cupy)tr_y: {tr_y.shape}")

    ts_x = cp.asarray(ts_x, dtype=cp.float16).reshape(10000, -1)
    print(f"[INFO]\t(cupy)ts_x: {ts_x.shape}")

    ts_y = ts_y.numpy()
    print(f"[INFO]\t(numpy)ts_y: {ts_y.shape}")
    
    return tr_x,tr_y,ts_x,ts_y

def load_torchFashionMnist():
    print("[INFO] loading torch Fashion MNIST (sample) dataset...")
    train_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
    )
    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
    )

    tr_x = train_data.data / 255
    tr_y = train_data.targets

    ts_x = test_data.data / 255
    ts_y = test_data.targets

    tr_x = cp.asarray(tr_x, dtype=cp.float16).reshape(60000, -1)
    print(f"[INFO]\t(cupy)tr_x:  {tr_x.shape}")

    tr_y = cp.asarray(tr_y, dtype=cp.int8)
    print(f"[INFO]\t(cupy)tr_y: {tr_y.shape}")

    ts_x = cp.asarray(ts_x, dtype=cp.float16).reshape(10000, -1)
    print(f"[INFO]\t(cupy)ts_x: {ts_x.shape}")

    ts_y = ts_y.numpy()
    print(f"[INFO]\t(numpy)ts_y: {ts_y.shape}")
    
    return tr_x,tr_y,ts_x,ts_y



def load_skMnist():
    print("[INFO] loading sklearn MNIST (sample) dataset...")
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())
    print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))
    (trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)
    # convert to onehot:
    # trainY = LabelBinarizer().fit_transform(trainY)
    # testY = LabelBinarizer().fit_transform(testY)

    trX = cp.asarray(trainX,dtype=cp.float16)
    print(f"[INFO]\t(cupy)tr_x:  {trX.shape}")
    trY = cp.asarray(trainY,dtype=cp.int8)
    print(f"[INFO]\t(cupy)tr_y: {trY.shape}")
    tsX = cp.asarray(testX,dtype=cp.float16)
    print(f"[INFO]\t(cupy)ts_x: {tsX.shape}")
    # tsY = cp.asarray(testY,dtype=cp.int32)
    tsY = testY
    print(f"[INFO]\t(numpy)ts_y: {tsY.shape}")
    
    return trX,trY,tsX,tsY


def load_Cifar100():
    print("[INFO] loading Cifar100 (sample) dataset...")
    train_data = torchvision.datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
    )
    test_data = torchvision.datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
    )

    tr_x = train_data.data / 255
    tr_y = train_data.targets
    
    ts_x = test_data.data / 255
    ts_y = test_data.targets

    tr_x = cp.asarray(tr_x.transpose(0,3,1,2), dtype=cp.float16)
    print(f"[INFO]\t(cupy)tr_x:  {tr_x.shape}")

    tr_y = cp.asarray(tr_y, dtype=cp.int8)
    print(f"[INFO]\t(cupy)tr_y: {tr_y.shape}")

    ts_x = cp.asarray(ts_x.transpose(0,3,1,2), dtype=cp.float16)
    print(f"[INFO]\t(cupy)ts_x: {ts_x.shape}")

    ts_y = np.asarray(ts_y, dtype=np.int8)
    print(f"[INFO]\t(numpy)ts_y: {ts_y.shape}")
    
    return tr_x,tr_y,ts_x,ts_y

def load_Cifar10():
    print("[INFO] loading Cifar10 (sample) dataset...")
    train_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
    )
    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
    )

    tr_x = train_data.data / 255
    tr_y = train_data.targets
    
    ts_x = test_data.data / 255
    ts_y = test_data.targets

    tr_x = cp.asarray(tr_x.transpose(0,3,1,2), dtype=cp.float16)
    print(f"[INFO]\t(cupy)tr_x:  {tr_x.shape}")

    tr_y = cp.asarray(tr_y, dtype=cp.int8)
    print(f"[INFO]\t(cupy)tr_y: {tr_y.shape}")

    ts_x = cp.asarray(ts_x.transpose(0,3,1,2), dtype=cp.float16)
    print(f"[INFO]\t(cupy)ts_x: {ts_x.shape}")

    ts_y = np.asarray(ts_y, dtype=np.int8)
    print(f"[INFO]\t(numpy)ts_y: {ts_y.shape}")
    
    return tr_x,tr_y,ts_x,ts_y


def load_SVHN():
    print("[INFO] loading SVHN (sample) dataset...")
    train_data = torchvision.datasets.SVHN(
        root="data",
        split="train",
        download=True,
    )
    test_data = torchvision.datasets.SVHN(
        root="data",
        split="test",
        download=True,
    )

    tr_x = train_data.data / 255
    tr_y = train_data.labels
    
    ts_x = test_data.data / 255
    ts_y = test_data.labels

    tr_x = cp.asarray(tr_x, dtype=cp.float16)
    print(f"[INFO]\t(cupy)tr_x:  {tr_x.shape}")

    tr_y = cp.asarray(tr_y, dtype=cp.int8)
    print(f"[INFO]\t(cupy)tr_y: {tr_y.shape}")

    ts_x = cp.asarray(ts_x, dtype=cp.float16)
    print(f"[INFO]\t(cupy)ts_x: {ts_x.shape}")

    ts_y = np.asarray(ts_y, dtype=np.int8)
    print(f"[INFO]\t(numpy)ts_y: {ts_y.shape}")
    
    return tr_x,tr_y,ts_x,ts_y