import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

from train import model
from create_dataset import test_dataset, test_loader


def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        data = Variable(data, volatile=True)
        if torch.cuda.is_available():
            data = data.cuda()

        output = model(data)

        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred


test_pred = prediciton(test_loader)

out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset) + 1)[:, None], test_pred.numpy()],
                      columns=['ImageId', 'Label'])

# Lưu kết quả đầu ra
out_df.to_csv('data/submission.csv', index=False)
