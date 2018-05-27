import torch.nn.functional as F
import torch
import numpy as np


class Trainer():

    def __init__(self, data_loader, model, criterion, optimizer,
                 use_cuda, num_epoch, save_epoch=10, clip=5):

        self.data_loader = data_loader
        self.model = model.train()
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epoch
        self.save_epoch = save_epoch
        self.clip = clip
        self.device = "cuda" if use_cuda else "cpu"

    def train(self):

        for epoch in range(0, self.num_epochs):
            self.optimizer.scheduler_step()

            running_loss = 0.0

            for i, batch in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                input_variable = batch['image'].to(self.device)

                input_variable.requires_grad_()
                target = batch['target'].to(self.device).view(-1)

                predict = self.model(input_variable)
                predict = F.sigmoid(predict).view(-1)

                loss = self.criterion(predict, target)
                loss.backward()
                self.optimizer.gradient_clip(self.clip)
                self.optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 999:
                    print('[%d], [%5d] loss: %.3f' %
                          (epoch, i + 1, running_loss / 1000))
                    running_loss = 0.0

            print('============================')
            print(str(epoch) + 'epoch:', loss.item())
            if epoch % self.save_epoch == self.save_epoch - 1:
                # self.save_model(self.model, self.optimizer)
                print(str(epoch) + 'validation score:' + str(self.validate()))

        return loss.item()

    def validate(self):
        with torch.no_grad():
            mse = 0.0
            for batch_i, batch in enumerate(self.data_loader):
                input_variable = batch['image'].to(self.device)
                predict = self.model(input_variable)
                predict = F.sigmoid(predict).view(-1)
                target = batch['target'].view(-1).to(self.device)
                mse += torch.sum((predict - target) ** 2).to("cpu").numpy()
            return np.sqrt(mse / batch['image'].size(0))

    def save_model(self, epoch):
        model_filename = '../output/save_point/' + \
            'model_' + str(epoch) + 'epochs.pth.tar'
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.model_optimizer.state_dict(),
        }
        torch.save(state, model_filename)
