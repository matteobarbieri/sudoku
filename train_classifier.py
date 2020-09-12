#!/usr/bin/env python
# coding: utf-8

# ## Train dataset on MNIST :(

from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as transforms

import torchvision
import torch


from tqdm import tqdm

from sacred import Experiment

from sacred.observers import MongoObserver

ex = Experiment("DELETEME-MNIST")

ex.observers.append(
    MongoObserver(
        url="mongodb://sample:password@localhost:27017/db?authSource=admin",
        db_name="db",
    )
)


@ex.config
def cfg():
    lr = 1e-4  # noqa
    batch_size = 256  # noqa
    n_epochs = 20  # noqa
    model = "resnet50"  # noqa


def get_model(model_name):
    if model_name == "resnet50":

        net = torchvision.models.resnet50(pretrained=True)

        # Replace 1st layer to use it on grayscale images
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    if model_name == "resnet101":

        net = torchvision.models.resnet101(pretrained=True)

        # Replace 1st layer to use it on grayscale images
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    return net


@ex.automain
def main(_run, lr, batch_size, n_epochs, model):

    train_transform = transforms.Compose([transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(
        "./",
        train=True,
        transform=train_transform,
        target_transform=None,
        download=True,
    )
    val_set = torchvision.datasets.MNIST(
        "./",
        train=False,
        transform=val_transforms,
        target_transform=None,
        download=True,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    device = "cuda"

    # Get network
    net = get_model(model)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    counter = 0

    for epoch in range(n_epochs):

        print(f"Running epoch {epoch+1}")

        running_loss = 0.0

        total_correct = 0
        total_processed = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

            net.train()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # Compute actual predictions
            sm = torch.nn.functional.softmax(outputs, dim=1)
            _, y_hat = torch.max(sm, 1)

            total_correct += (labels == y_hat.to("cpu")).sum().item()
            total_processed += len(labels)

            counter += 1

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                _run.log_scalar("training.loss", running_loss / 100, counter)
                _run.log_scalar(
                    "training.acc",
                    float(total_correct) / total_processed,
                    counter,
                )
                running_loss = 0.0

                # Check validation set
                net.eval()

                with torch.no_grad():

                    total_processed_val = 0
                    total_correct_val = 0

                    for data in val_loader:
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data

                        # forward + backward + optimize
                        outputs = net(inputs.to(device))
                        loss = criterion(outputs, labels.to(device))

                        # Compute actual predictions
                        sm = torch.nn.functional.softmax(outputs, dim=1)
                        _, y_hat = torch.max(sm, 1)

                        total_processed_val += len(labels)
                        total_correct_val += (
                            (labels == y_hat.to("cpu")).sum().item()
                        )

                        running_loss += loss.item()

                    _run.log_scalar(
                        "validation.loss", running_loss / 100, counter
                    )

                    _run.log_scalar(
                        "validation.acc",
                        float(total_correct_val) / total_processed_val,
                        counter,
                    )

                    running_loss = 0.0

        # Save model at the end of every epoch
        torch.save(net.state_dict(), f"./{model}_mnist.pth")

    print("Finished Training")
