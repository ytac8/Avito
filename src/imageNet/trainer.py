import torch.nn.functional as F
import torch
import copy
import numpy as np


class Trainer():

    def __init__(self, train_loader, val_loader, model, criterion, optimizer,
                 use_cuda, num_epoch, save_epoch=15, clip=5, valid_interval=1000):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.train()
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = clip
        self.device = "cuda" if use_cuda else "cpu"

        # validation系
        self.best_model = None
        self.best_mse = np.inf
        self.best_iter = -1
        self.valid_interval = valid_interval
        self.num_epochs = num_epoch
        self.save_epoch = save_epoch

    def train(self):

        for epoch in range(0, self.num_epochs):
            if epoch % 10 == 9:
                self.optimizer.scheduler_step()

            running_loss = 0.0

            for i, batch in enumerate(self.train_loader):
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
                if i % self.valid_interval == self.valid_interval - 1:
                    # val_score = self.validate()
                    val_score = running_loss / self.valid_interval
                    print('[%d], [%5d] loss: %.3f validataion score: %.5f' %
                          (epoch, i + 1, running_loss / self.valid_interval, val_score))

                    if val_score < self.best_mse:
                        self.best_model = copy.deepcopy(self.model.state_dict)
                        self.best_mse = val_score
                        self.best_iter = i
                    running_loss = 0.0

            print('============================')
            print(str(epoch) + 'epoch:', loss.item())
            if epoch % self.save_epoch == self.save_epoch - 1:
                val_score = self.validate()
                print(str(epoch) + 'validation score:' + str(val_score))
                self.save_model(epoch, self.best_iter, val_score)

        return loss.item()

    def validate(self):
        with torch.no_grad():
            mse = 0.0
            for batch_i, batch in enumerate(self.val_loader):
                input_variable = batch['image'].to(self.device)
                predict = self.model(input_variable)
                predict = F.sigmoid(predict).view(-1)
                target = batch['target'].view(-1).to(self.device)
                mse += torch.sum((predict - target) ** 2).to("cpu").numpy()
            return np.sqrt(mse / len(self.val_loader))

    def save_model(self, epoch, best_iter, val_score):
        model_filename = '../output/save_point/' + \
            'model_' + str(epoch) + '_epoch_iter_' + str(best_iter) + \
            '_val' + str(val_score) + '.pth.tar'
        state = {
            'state_dict': self.best_model
        }
        torch.save(state, model_filename)
