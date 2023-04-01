
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_module import Train
from valid_module import Valid
from load_module import Load
from model import VGG16


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("PyTorch 版本: ", torch.__version__)
    print("CUDA 版本: ", torch.version.cuda)
    print("cuDNN 版本: ", str(torch.backends.cudnn.version()))
    print("设备名称: ", torch.cuda.get_device_name(0))

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 8

    load = Load()
    train_loader, valid_loader = load.load_data("./datasets", 64, 100, num_workers)

    model = VGG16(init_weights=True)
    model.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train = Train()
    valid = Valid()

    epochs = 100

    min_loss = 1.0
    max_loss = 0.0

    min_train_acc = 1.0
    max_train_acc = 0.0

    best_train_acc = 0.0
    best_valid_acc = 0.0
    min_valid_acc = 1.0
    max_valid_acc = 0.0

    Loss = []
    Train_Acc = []
    Valid_Acc = []

    model_path = "./model/VGG16-CIFAR10.pth"
    if not os.path.exists("./model"):
        os.mkdir("./model")

    img_path = "./img/VGG16-CIFAR10.jpg"
    if not os.path.exists("./img"):
        os.mkdir("./img")

    print("start-time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    for epoch in range(epochs + 1):
        loss, train_acc = train.train_method(model, device, train_loader, optimizer, loss_func, epoch)

        if loss < min_loss:
            min_loss = loss
        if loss > max_loss:
            max_loss = loss

        Loss.append(loss)

        if train_acc < min_train_acc:
            min_train_acc = train_acc
        if train_acc > max_train_acc:
            max_train_acc = train_acc
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        Train_Acc.append(train_acc)

        valid_acc = valid.valid_method(model, device, valid_loader, epoch)

        if valid_acc < min_valid_acc:
            min_valid_acc = valid_acc
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_path)

        Valid_Acc.append(valid_acc)

    print("end-time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    plt.figure(figsize=(5, 7))

    plt.subplot(3, 1, 1)
    plt.plot(Loss)
    plt.title("Loss")
    plt.xticks(torch.arange(0, epochs+1, 10))
    plt.yticks(torch.arange(min_loss, max_loss + 0.3, 0.3))

    plt.subplot(3, 1, 2)
    plt.plot(Train_Acc)
    plt.title("Train_Acc")
    plt.xticks(torch.arange(0, epochs + 1, 10))
    plt.yticks(torch.arange(min_train_acc, max_train_acc, 0.2))
    train_label = "best train acc: " + str(best_train_acc)
    plt.xlabel(train_label)

    plt.subplot(3, 1, 3)
    plt.plot(Valid_Acc)
    plt.title("Valid_Acc")
    plt.xticks(torch.arange(0, epochs + 1, 10))
    plt.yticks(torch.arange(min_valid_acc, max_valid_acc, 0.2))
    valid_label = "best valid acc: " + str(best_valid_acc)
    plt.xlabel(valid_label)

    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1, hspace=0.55)
    plt.savefig(img_path)
    plt.show()


if __name__ == "__main__":
    main()
