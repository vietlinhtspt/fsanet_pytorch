from torch.utils.data import Dataset

class BIWIDataset(Dataset):
    def __init__(self, base_dir=None):
        print("[INFO] Initing BIWIDataset.")
        