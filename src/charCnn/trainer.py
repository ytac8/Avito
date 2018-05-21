class Trainer():

    def __init__(self, data_loader, model, criterion, optimizer,
                 use_cuda, clip=5):

        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = clip
        self.device = "cuda" if use_cuda else "cpu"

    def train(self):

        self.optimizer.scheduler_step()

        running_loss = 0.0
        for i, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            input_variable = batch['feature'].to(self.device)
            target = batch['label'].to(self.device).view(-1)
            predict = self.model(input_variable).view(-1)

            loss = self.criterion(predict, target)
            loss.backward()
            self.optimizer.gradient_clip(self.clip)
            self.optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:    # print every 100 mini-batches
                print('[%5d] loss: %.3f' %
                      (i + 1, running_loss / 1000))
                running_loss = 0.0

        return loss.item()
