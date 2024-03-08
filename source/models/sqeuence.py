from .fbnetgen import ConvKRegion, GruKRegion
import torch.nn as nn
from torch.nn import Linear
from .base import BaseModel


class SeqenceModel(BaseModel):

    def __init__(self, config, roi_num=360, time_series=512):
        super().__init__()

        if config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=config['embedding_size'], kernel_size=config['window_size'],
                time_series=time_series, pool_size=4, )
        elif config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=config['embedding_size'], kernel_size=config['window_size'],
                layers=config['num_gru_layers'], dropout=config['dropout'])

        output_dim = 1 if config.dataset.regression else config.dataset.num_classes
        self.linear = nn.Sequential(
            Linear(config['embedding_size']*roi_num, 256),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            Linear(256, 32),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            Linear(32, output_dim),
        )

    def forward(self, time_seires, node_feature):
        x = self.extract(time_seires)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x
