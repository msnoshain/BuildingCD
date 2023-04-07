import time
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class DataTrainer():
    dataset: data.Dataset
    module: nn.Module

    batch_size: int = 1
    optimizer: optim.Optimizer
    loss_function: nn.Module = nn.BCELoss()
    epochs: int = 200
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, dataset: data.Dataset, module: nn.Module):
        self.dataset = dataset
        self.module = module
        
        if self.module is not None:
            self.optimizer = optim.Adam(self.module.parameters())

    def train(self):
        if self.dataset is None or self.module is None:
            raise ValueError(self.dataset, self.module)

        self.module.to(device=self.device)

        dataloaders = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=2 if self.is_using_cuda() else 0,
                                      pin_memory=self.is_using_cuda())
 
        for e in range(self.epochs):
            print('Epoch {}/{} begins at {}'.format(e + 1, self.epochs, time.strftime("%H:%M:%S",time.localtime())))
            epoch_loss = 0

            for x, y in dataloaders:
                inputs = x.to(self.device)
                labels = y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.module(inputs)
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss)

            print("loss:%0.3f" % (epoch_loss))
            print('-' * 10)

    def is_using_cuda(self):
        if self.device.type is "cuda":
            return True

        return False
