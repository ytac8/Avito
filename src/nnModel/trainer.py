import torch
import copy
import numpy as np


class Trainer():

    def __init__(self, train_loader, val_loader, model, criterion, optimizer,
                 use_cuda, num_epoch, save_epoch=15, clip=10,
                 valid_interval=500):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.train()
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # validation系
        self.best_model = None
        self.best_mse = np.inf
        self.best_iter = -1
        self.valid_interval = valid_interval
        self.num_epochs = num_epoch
        self.save_epoch = save_epoch

    def train(self):

        for epoch in range(0, self.num_epochs):
            self.optimizer.scheduler_step()

            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                input_variable = batch['feature'].to(self.device).squeeze()
                target = batch['target'].to(self.device).squeeze()
                predict = self.model(input_variable)

                loss = self.criterion(predict, target)
                loss.backward()
                self.optimizer.gradient_clip(self.clip)
                self.optimizer.step()

            val_score = self.validate(self.model.eval())
            if val_score < self.best_mse:
                self.best_model = copy.deepcopy(self.model.state_dict)
                self.best_mse = val_score
                self.best_iter = i

            self.model.train()

            print('============================')
            print(str(epoch) + 'epoch:', loss.item())
            print(str(epoch) + 'validation score:' + str(val_score))

            if epoch % self.save_epoch == self.save_epoch - 1:
                self.save_model(epoch, self.best_iter, val_score)

        return loss.item()

    def validate(self, model):
        mse = 0.0
        dataset_length = 0
        with torch.no_grad():
            for batch_i, batch in enumerate(self.val_loader):
                dataset_length += batch['target'].size(0)
                input_variable = batch['feature'].to(self.device).squeeze()
                target = batch['target'].to(self.device).squeeze().float()
                predict = model(input_variable)

                predict = predict[:, 1]
                mse += torch.sum((predict - target) **
                                 2).to("cpu").detach().numpy()
            return np.sqrt(mse / dataset_length)

    def save_model(self, epoch, best_iter, val_score):
        model_filename = './output/save_point/' + \
            'model_' + str(epoch) + '_epoch_iter_' + str(best_iter) + \
            '_val' + str(val_score) + '.pth.tar'
        state = {
            'state_dict': self.best_model
        }
        torch.save(state, model_filename)
