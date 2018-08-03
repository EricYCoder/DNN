import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm


class Trainer(object):
    def __init__(self, net, cuda, model_name):
        self.net = net
        self.model_name = model_name
        if cuda:
            self.cuda = True
            self.device = torch.device("cuda")
            print("GPU")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
            print("CPU")

    def predict(self, data):
        """
        data:np.array
        """
        data = torch.tensor(data, dtype=torch.float32)

        if self.cuda:
            data = data.to(self.device)

        with torch.no_grad():
            pred = self.net(data)

        _, pred = torch.max(pred, dim=1)

        if self.cuda:
            pred = pred.cpu()

        pred = pred.numpy()
        return pred

    def train_model(
        self,
        train_dataset,
        test_dataset,
        batch_size=32,
        epoch=300,
        optimizer_func="SGD",
        lr_rate=1e-4,
    ):

        if self.cuda:
            self.net = self.net.to(self.device)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )

        self.net.train()
        if optimizer_func is "Adam":
            optimizer = optim.Adam(self.net.parameters(), lr=lr_rate)
        elif optimizer_func is "Adagrad":
            optimizer = optim.Adagrad(self.net.parameters(), lr=lr_rate)
        elif optimizer_func is "RMSprop":
            optimizer = optim.RMSprop(self.net.parameters(), lr=lr_rate)
        else:
            optimizer = optim.SGD(self.net.parameters(), lr=lr_rate)

        criterion = nn.CrossEntropyLoss()

        for e in range(epoch):
            for i, sample in tqdm.tqdm(enumerate(train_loader)):

                optimizer.zero_grad()

                spectral = sample["spectral"]
                spectral = spectral.to(torch.float32)
                label = sample["label"]
                label = label.to(torch.long)
                if self.cuda:
                    spectral = spectral.to(self.device)
                    label = label.to(self.device)
                pred = self.net(spectral)
                loss = criterion(pred, label)

                loss.backward()
                optimizer.step()

            # every epoch, compute the accuray
            _, pred = torch.max(pred, dim=1)
            label = label.view(-1)
            accuracy = torch.eq(pred, label).type(torch.FloatTensor)
            accuracy = torch.sum(accuracy)
            accuracy = torch.div(accuracy, batch_size)

            accuracy = accuracy.cpu()
            loss = loss.cpu()
            print(
                "Train-Epoch:{}/{},accuracy:{}, loss:{}".format(
                    (e + 1), epoch, accuracy, loss
                )
            )

            if (e + 1) % 10 == 0:
                self.net.eval()
                loss = 0
                accuracy = 0
                for j, sample in enumerate(test_loader, 0):
                    spectral = sample["spectral"]
                    spectral = spectral.to(torch.float32)
                    label = sample["label"]
                    label = label.to(torch.long)
                    if self.cuda:
                        spectral = spectral.to(self.device)
                        label = label.to(self.device)
                    with torch.no_grad():
                        pred = self.net(spectral)
                    loss = loss + criterion(pred, label)
                    _, pred = torch.max(pred, dim=1)
                    tmp_accuracy = torch.eq(pred, label).type(torch.FloatTensor)
                    tmp_accuracy = torch.div(tmp_accuracy.sum(), len(label))
                    accuracy = accuracy + tmp_accuracy
                accuracy = torch.div(accuracy, len(test_loader))
                loss = torch.div(loss, len(test_loader))
                self.net.train()

                accuracy = accuracy.cpu()
                loss = loss.cpu()
                print(
                    "Test-Epoch:{}/{},accuracy:{}, loss:{}".format(
                        (e + 1), epoch, accuracy, loss
                    )
                )
                self.save_model()

    def save_model(self):
        if self.cuda:
            self.net = self.net.cpu()
        torch.save(self.net, self.model_name)
        if self.cuda:
            self.net = self.net.to(self.device)
        print("model saved!")

    def restore_model(self):
        self.net = torch.load(self.model_name)
        if self.cuda:
            self.net = self.net.to(self.device)
        print("model restored!")
