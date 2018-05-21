import pandas as pd
import torch


class Predictor():

    def __init__(self, data_loader, model, cuda, item_id_dict, is_train=True):
        self.data_loader = data_loader
        self.device = torch.device("cuda" if cuda else "cpu")
        self.item_id_dict = {v: k for k, v in item_id_dict.items()}
        self.is_train = is_train
        self.model = model.eval()
        if not is_train:
            checkpoint = torch.load(
                '../output/save_point/model_10epochs.pth.tar')
            self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self):
        with torch.no_grad():
            item_id_list = []
            pred_list = []
            target_list = [] if self.is_train else None

            for batch_i, batch in enumerate(self.data_loader):
                input_variable = batch['feature'].to(self.device)
                item_ids = self.item_id_decode(
                    batch['item_id'].view(-1).tolist())
                predict = self.model(input_variable)
                if self.is_train:
                    target = batch['label'].view(-1).tolist()
                    target_list.extend(target)

                item_id_list.extend(item_ids)
                pred_list.extend(predict.view(-1).tolist())

            return pred_list, item_id_list, target_list

    def item_id_decode(self, item_id_list):
        item_id_list = [self.item_id_dict[x] for x in item_id_list]
        return item_id_list

    def output_prediction(self):
        item_id = pd.Series(self.item_id_list)
        deal_probability = pd.Series(self.pred_list)
        submission_df = pd.DataFrame()
        submission_df['item_id'] = item_id
        submission_df['deal_probability'] = deal_probability
        return submission_df
