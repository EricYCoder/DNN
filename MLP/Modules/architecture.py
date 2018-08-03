import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_num=3):
        super(MLP, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, out_channels),
            nn.Softmax(dim=1),
        )

        """
        self.fc_in = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        if hidden_num >= 4:
            self.fc_mid = nn.Sequential(
                nn.Linear(256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout()
            )
            self.fc_in = nn.Sequential(self.fc_in, self.fc_mid)

        self.fc_mid = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout()
        )
        self.fc_in = nn.Sequential(self.fc_in, self.fc_mid)

        if hidden_num >= 5:
            self.fc_mid = nn.Sequential(
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout()
            )
            self.fc_in = nn.Sequential(self.fc_in, self.fc_mid)

        self.fc_out = nn.Sequential(nn.Linear(128, out_channels), nn.Softmax(dim=1))

        self.main = nn.Sequential(self.fc_in, self.fc_out)
        """

    def forward(self, x):
        res = self.main(x)
        return res
