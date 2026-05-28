import lightning as L
import torch
import numpy as np
from .transforms import PendulumTruncateTransform, FingerTruncateTransform, ShallowWaterTruncateTransform
from .datasets import PendulumDataset, FingerDataset, ShallowWaterDataset

class PendulumReconDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        batch_size=32,
        test_batch_size=32,
        val_fraction=0.10,
        test_fraction=0.10,
        num_workers=0,
        seed=42                    
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        full_ds = PendulumDataset(self.train_data_path, transform=None)  
        n_total = len(full_ds)
        n_val = max(1, int(round(self.val_fraction * n_total)))
        n_test = max(1, int(round(self.test_fraction * n_total)))
        n_train = n_total - n_val - n_test
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_total-n_test]
        test_idx = perm[n_train+n_val:]
        
        if stage in (None, "fit"):
            self.train_dataset = torch.utils.data.Subset(full_ds, train_idx)
            self.val_dataset = torch.utils.data.Subset(full_ds, val_idx)

        if stage in (None, "test"):
            self.test_dataset = torch.utils.data.Subset(full_ds, test_idx)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

class PendulumExtrapDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        batch_size=32,
        test_batch_size=32,
        val_split=0.6,
        test_split=0.8,
        num_workers=0
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.train_transform = PendulumTruncateTransform(self.val_split)
        self.val_transform = PendulumTruncateTransform(self.test_split)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = PendulumDataset(self.train_data_path, self.train_transform) # truncate to training portion
            self.val_dataset = PendulumDataset(self.train_data_path, self.val_transform) # truncate to training + val portion
        elif stage == "test":
            self.test_dataset = PendulumDataset(self.train_data_path) # test on full sequence

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,    
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True
        )
    def val_dataloader(self):
           return torch.utils.data.DataLoader(
            self.val_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False
        )
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        
class FingerReconDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        batch_size=32,
        test_batch_size=32,
        val_fraction=0.10,
        test_fraction=0.10,
        num_workers=0,
        seed=42                    
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        full_ds = FingerDataset(self.train_data_path, transform=None)  
        n_total = len(full_ds)
        n_val = max(1, int(round(self.val_fraction * n_total)))
        n_test = max(1, int(round(self.test_fraction * n_total)))
        n_train = n_total - n_val - n_test
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_total-n_test]
        test_idx = perm[n_train+n_val:]
        
        if stage in (None, "fit"):
            self.train_dataset = torch.utils.data.Subset(full_ds, train_idx)
            self.val_dataset = torch.utils.data.Subset(full_ds, val_idx)

        if stage in (None, "test"):
            self.test_dataset = torch.utils.data.Subset(full_ds, test_idx)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

class FingerExtrapDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        batch_size=32,
        test_batch_size=32,
        val_split=0.6,
        test_split=0.8,
        num_workers=0
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.train_transform = FingerTruncateTransform(self.val_split)
        self.val_transform = FingerTruncateTransform(self.test_split)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = FingerDataset(self.train_data_path, self.train_transform) # truncate to training portion
            self.val_dataset = FingerDataset(self.train_data_path, self.val_transform) # truncate to training + val portion
        elif stage == "test":
            self.test_dataset = FingerDataset(self.train_data_path) # test on full sequence

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,    
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True
        )
    def val_dataloader(self):
           return torch.utils.data.DataLoader(
            self.val_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False
        )
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False
        )