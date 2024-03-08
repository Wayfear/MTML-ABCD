import imp
from .transformer import GraphTransformer
from omegaconf import DictConfig
from .EGT import EGT
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .BNT import BrainNetworkTransformer
from .braingnn import BrainGNN
from .sparsenn import SparseNN
from .MATT import MixOptimizer, MATT
from .mtmlmodel import MTMLBNTWGaitM


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC", "ElasticNet", "GradientBoostingRegressor", "RandomForestClassifier"]:
        return None
    return eval(config.model.name)(config).cuda()
