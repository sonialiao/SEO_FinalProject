from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import onnx
import onnxruntime as ort

from dataset import get_train_test_loaders
from train import Net

def evaluate(outputs, labels) -> float:
    
    # numpy array of actual labels
    Y = labels.numpy()  

    # convert model output to prediction
    Y_hat = np.argmax(outputs, axis=1) 

    return float(np.sum(Y_hat == Y))


# evaluate every "batch" of images from dataloader
def batch_evaluate(net, dataloader):
    score = 0.0
    n = 0.0

    for batch in dataloader:
        n += len(batch['image'])

        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()

        # labels are in column 0 under batch['label']
        score += evaluate(outputs, batch['label'][:, 0])

    return score / n


# take data loaders and feed train and test set into batch_eval
def validate():

    # load in datasets & model (after being trained in train.py)
    trainloader, testloader = get_train_test_loaders()
    net = Net().float()
    pretrained_model = torch.load("checkpoint.pth")
    net.load_state_dict(pretrained_model)

    print('=' * 10, 'PyTorch', '=' * 10)

    train_accuracy = batch_evaluate(net, trainloader) * 100.
    print(f"Training accuracy: {train_accuracy:.1f}")
    test_accuracy = batch_evaluate(net, testloader) * 100.
    print(f"Validation accuracy: {test_accuracy:.1f}")


# export model to ONNX format and re-validate accuracy
def export():

    # load in dataset as 1 batch: no need for smaller batchs w/o training
    trainloader, testloader = get_train_test_loaders(1)

    # build model again
    net = Net().float()
    pretrained_model = torch.load("checkpoint.pth")
    net.load_state_dict(pretrained_model)

    # export to onnx
    fname = "signlanguage.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])

    # check exported model as well-formed
    model = onnx.load(fname)
    onnx.checker.check_model(model)

    # create runnable session with exported model
    ort_session = ort.InferenceSession(fname)
    net = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]

    # validate ONNX session
    print('=' * 10, 'ONNX', '=' * 10)
    train_accuracy = batch_evaluate(net, trainloader) * 100.
    print(f"Training accuracy: {train_accuracy:.1f}")
    test_accuracy = batch_evaluate(net, testloader) * 100.
    print(f"Validation accuracy: {test_accuracy:.1f}")


if __name__ == '__main__':
    validate()
    export()