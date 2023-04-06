import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from extentions.utils import utils


class DataTrainer():
    dataset: data.Dataset
    module: nn.Module

    batch_size: int = 4
    optimizer: optim.Optimizer = optim.Adam
    loss_function: nn.Module = nn.BCELoss
    epochs: int = 200
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, dataset: data.Dataset, module: nn.Module):
        self.dataset = dataset
        self.module = module
        self.optimizer.add_param_group(self.module.parameters())

    def train(self):
        if self.dataset is None or self.module is None:
            raise ValueError(self.dataset, self.module)

        dataloaders = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4 if utils.is_using_cuda(self.device) else 0, 
                                      pin_memory=utils.is_using_cuda(self.device))

        for e in range(self.epochs):
            print('Epoch {}/{}'.format(e, self.epochs - 1))
            print('-' * 10)
            dt_size = len(dataloaders.dataset)
            epoch_loss = 0
            step = 0
            print(dt_size)
            running_loss = 0
            index = 0

            for x, y in dataloaders:
                step += 1
                inputs = x.to(self.device)
                labels = y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.module(inputs)
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                running_loss += loss.item()
                index += 1

                if index % 50 == 49:
                    print("Epoch:%d Batch:%d Loss is %f" %
                          (e + 1, index + 1, running_loss / 50))
                    running_loss = 0.0

                print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) //
                      dataloaders.batch_size + 1, loss.item()))

            print("epoch %d loss:%0.3f" % (e, epoch_loss))
