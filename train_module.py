
import sys
import torch
from tqdm import tqdm
import torch.utils.data as data
import torch.optim as opt
from model import VGG16


class Train:
    def train_method(self, model: VGG16, device: torch.device, train_loader: data.DataLoader, optimizer: opt,
                     loss_func, epoch: int):

        model.train()
        model = model.to(device)

        total = 0.0
        correct = 0.0
        sum_loss = 0.0
        num = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            num = step + 1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            sum_loss += loss.item()
            predicts = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += torch.eq(predicts, labels).sum().item()
            loss.backward()
            optimizer.step()
            train_bar.desc = "Train Epoch: {:d}, Loss: {:.3f}, Acc: {:.3f} %".format(epoch, sum_loss / num, 100 * (correct / total))

        return round(sum_loss / num, 3), round(correct / total, 3)
