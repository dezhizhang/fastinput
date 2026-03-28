import pandas as pd
import torch
from torch.utils.data import Dataset
import config
from torch.utils.data import DataLoader


class FastInputDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['target'], dtype=torch.long)

        return input_tensor, target_tensor


def get_dataloader(train=True):
    """获取数据集方法"""
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = FastInputDataset(path)

    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))
