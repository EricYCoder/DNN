import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import settings


def prepare_data(file_path):
    dataset = np.load(file_path)
    features = dataset["features"]
    labels = dataset["labels"]
    labels = labels[:, np.newaxis]
    dataAll = np.concatenate([features, labels], axis=1)
    idx = np.arange(len(dataAll))
    np.random.shuffle(idx)
    dataAll = dataAll[idx, :]

    data = []
    recordNum = len(labels)
    target_num = [
        recordNum / 9,
        recordNum / (9 * 6),
        recordNum / 9,
        recordNum / (9 * 6),
        recordNum / (9 * 6),
        recordNum / (9 * 6),
        recordNum * 2 / (9 * 6),
    ]
    data_num = [0, 0, 0, 0, 0, 0, 0]
    # change other labels to 10
    for index in range(len(dataAll)):
        if data_num[int(dataAll[index][-1])] >= target_num[int(dataAll[index][-1])]:
            continue
        else:
            data_num[int(dataAll[index][-1])] += 1
            if dataAll[index][-1] not in [0, 2]:
                dataAll[index][-1] = 1
            data.append(list(dataAll[index]))

    data = np.array(data)

    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx, :]

    return data


class Spectrum(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        size = len(self.data[index]) - 1
        spectral = self.data[index][0:size]
        label = self.data[index][-1]
        sample = {}
        sample["spectral"] = spectral
        sample["label"] = label
        return sample
