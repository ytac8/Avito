import torch
import torch.optim as optim
from torch.optim import lr_scheduler


class Optimizer():

    def __init__(self, model, parameters, lr=0.1):
        self.model = model
        self.model_optimizer = optim.SGD(
            parameters, lr=lr, momentum=0.9)
        self.model_scheduler = lr_scheduler.StepLR(
            self.model_optimizer, step_size=5, gamma=0.99)

    def zero_grad(self):
        self.model_optimizer.zero_grad()

    def step(self):
        self.model_optimizer.step()

    def scheduler_step(self):
        self.model_scheduler.step()

    def gradient_clip(self, clip):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
