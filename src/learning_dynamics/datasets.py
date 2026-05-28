import pickle
import torch

class PendulumDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.thetas = data["thetas"] # num_samples, num_timepoints, 1
            self.winds = data["winds"]
            self.frames = data["frames"] # num_samples, num_frames, px, py
            self.time = data["time"] # 10
            self.dt = data["dt"] # 0.01
            
        _, _, self.height, self.width = self.frames.shape
        self.transform = transform

    def __getitem__(self, index):
        frames = torch.tensor(self.frames[index]).double()
        thetas = torch.tensor(self.thetas[index]).double()
        if self.transform is not None: 
            return self.transform((frames, thetas))
        return frames, thetas

    def __len__(self):
        return self.thetas.shape[0]
        
class FingerDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.frames = data["frames"] # num_samples, num_frames, px, py
            self.latents = data["latents"]
            self.time = data["time"] # 10
            self.dt = data["dt"] # 0.01
            self.subsample = data["subsample"]
            self.width = data["width"]
            self.height = data["height"]
            
        self.transform = transform

    def __getitem__(self, index):
        frames = torch.tensor(self.frames[index]).double().div_(255.0)
        latents = torch.tensor(self.latents[index]).double()
        if self.transform is not None: 
            return self.transform((frames, latents))
        return frames, latents

    def __len__(self):
        return self.frames.shape[0]