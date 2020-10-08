from torch.utils.data import Dataset

class ALFW2000Dataset(Dataset):
    def __init__(self, base_dir=None):
        print("[INFO] Initing ALFW2000Dataset.")
