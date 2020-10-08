from .biwi import BIWIDataset
from .aflw2000 import AFLW2000Dataset
from .rank_300w_lp import Rank300WLPDataset

def load_dataset(data_type="300WLP", **kwargs):
    if data_type == "BIWI":
        return BIWIDataset(**kwargs)
    elif data_type == "AFLW2000":
        return AFLW2000Dataset(**kwargs)
    elif data_type == "300W_LP":
        return Rank300WLPDataset(**kwargs)
    else:
        raise NotImplementedError