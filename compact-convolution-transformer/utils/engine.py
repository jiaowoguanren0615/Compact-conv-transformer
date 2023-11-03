import sys
import torch
from tqdm import tqdm
import torch.nn as nn
from torchinfo import summary


def train_step(net, optimizer, lr_scheduler, data_loader, device, epoch, scalar=None):
    net.train()
    loss_function = nn.CrossEntropyLoss()
    train_acc, train_loss, sampleNum = 0, 0, 0
    optimizer.zero_grad()

    train_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(train_bar):
        images, labels = data
        sampleNum += images.shape[0]  # batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scalar is not None:
            with torch.cuda.amp.autocast():
                outputs = net(images)
                loss = loss_function(outputs, labels)
        else:
            outputs = net(images)
            loss = loss_function(outputs, labels)

        train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        train_loss += loss.item()
        # loss.backward()
        # optimizer.step()

        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        train_bar.desc = "[train epoch {}], loss: {:.3f}, acc: {:.3f}, lr: {:.3f}".format(epoch, train_loss / (step + 1),
                                                                             train_acc / sampleNum, lr)

    return train_loss / (step + 1), train_acc / sampleNum


@torch.no_grad()
def val_step(net, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss()
    net.eval()
    val_acc = 0
    val_loss = 0
    sample_num = 0
    val_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(val_bar):
        images, labels = data
        sample_num += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        val_bar.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, val_loss / (step + 1),
                                                                           val_acc / sample_num)

    return val_loss / (step + 1), val_acc / sample_num


def get_net_summary(net, input_size):
    return summary(net, input_size, col_names=["input_size",
                                               "output_size",
                                               "num_params",
                                               "params_percent",
                                               "kernel_size",
                                               "mult_adds",
                                               "trainable"])