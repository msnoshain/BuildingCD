import time
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class ModuleTrainer():
    """
    This is a class for training a model. 
    Simply deliver the model you want to train and the 
    dataset you'll use. Then set some training options 
    and train the model.
     
    Attributes:
        dataset:        the dataset that will be used to train the model
        module:         the module that will be trained
        batch_size:     batch size
        optimizer:      optimizer
        loss_function:  loss funtion
        epoch:          epoch
        device:         device that will be used to train the model
        save_frequency: determine the calculation graph will be saved every how many epochs
    """

    dataset: data.Dataset
    module: nn.Module

    batch_size: int = 1
    optimizer: optim.Optimizer
    loss_function: nn.Module = nn.BCELoss()
    epoch: int = 200
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    save_frequency: int = 0
    pt_path: str = None

    def __init__(self,
                 dataset: data.Dataset = None,
                 module: nn.Module = None,
                 batch_size: int = 2,
                 loss_function: nn.Module = nn.BCELoss(),
                 epoch: int = 200,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 save_frequency: int = 0,
                 pt_path: str = None):

        if dataset is None:
            raise ValueError(dataset)

        if module is None:
            raise ValueError(module)

        self.dataset = dataset
        self.module = module
        self.module_name = module.__class__.__name__

        self.batch_size = batch_size
        self.loss_function = loss_function
        self.epoch = epoch
        self.device = device
        self.optimizer = optim.Adam(self.module.parameters())

        self.save_frequency = save_frequency
        self.pt_path = pt_path

    def train(self):
        # params checking
        if self.dataset is None or self.module is None:
            raise ValueError(self.dataset, self.module)

        # move data to target device
        self.module.to(device=self.device)

        dataloader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=2,
                                     pin_memory=self.is_using_cuda())

        # train
        for e in range(self.epoch):
            # print info per epoch
            print('Epoch {}/{} begins at {}'.format(e + 1, self.epoch,
                  time.strftime("%H:%M:%S", time.localtime())))

            # calculate
            epoch_loss = 0
            for x, y in dataloader:
                inputs = x.to(self.device)
                labels = y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.module(inputs)
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss)

            # report loss
            print("loss:%0.3f" % (epoch_loss))
            print('-' * 10)

            # determine whether saving the calculation graph
            if self.save_frequency > 0 and (e+1) % self.save_frequency is 0:
                if self.pt_path is None:
                    raise ValueError(self.pt_path)

                torch.save(self.module, self.pt_path +
                           "\{}_{}.pt".format(self.module_name, e+1))

    def is_using_cuda(self):
        if self.device.type is "cuda":
            return True

        return False
