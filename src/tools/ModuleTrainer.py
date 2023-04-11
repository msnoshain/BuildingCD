import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


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
        save_frequency: determine the computational graph will be saved every how many epochs
    """

    save_frequency: int = 0
    pt_path: str = None

    current_loss: float

    def __init__(self,
                 dataset: data.Dataset = None,
                 module: nn.Module = None,
                 # Choose proper batch_size if you are using GPU.
                 # For unet with LEVIR-CD，GPU_RAM should >= batch_size * 4GB
                 batch_size: int = 1,
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

    def report_loss(loss: float, epoch: int):
        pass

    def train(self):
        # params checking
        if self.dataset is None or self.module is None:
            raise ValueError(self.dataset, self.module)

        # move module to target device
        self.module.to(device=self.device)

        # num_workers一般设为CPU的核心数，会降低CPU占用
        # batch_size适当提高可以增快收敛速度，但会占据更高的显存
        dataloader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=os.cpu_count(), pin_memory=False)

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

                outputs = self.module(inputs)
                loss = self.loss_function(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss)

            # report loss
            print("loss:%0.3f" % (epoch_loss))
            print('-' * 10)
            
            self.report_loss(epoch_loss, e + 1)

            # determine whether saving the computational graph
            if self.save_frequency > 0 and (e+1) % self.save_frequency is 0:
                if self.pt_path is None:
                    raise ValueError(self.pt_path)

                torch.save(self.module, self.pt_path +
                           "\\{}_ep{}_loss%0.3f.pt".format(self.module_name, e+1) % (epoch_loss))
