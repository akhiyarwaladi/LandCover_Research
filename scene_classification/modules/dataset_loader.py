"""
Dataset Loader for Remote Sensing Scene Classification Benchmarks

Supports: EuroSAT, NWPU-RESISC45, AID
"""

import os
import glob
import shutil
import zipfile
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, DATASET_PATHS, TRAINING, AUGMENTATION


class SceneDataset(Dataset):
    """Generic scene classification dataset."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_train_transform(image_size=None):
    """Training augmentation pipeline."""
    size = image_size or TRAINING['image_size']
    aug = AUGMENTATION
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(aug['random_rotation']),
        transforms.ColorJitter(
            brightness=aug['color_jitter_brightness'],
            contrast=aug['color_jitter_contrast'],
            saturation=aug['color_jitter_saturation'],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=aug['normalize_mean'], std=aug['normalize_std']),
    ])


def get_test_transform(image_size=None):
    """Test/validation transform (no augmentation)."""
    size = image_size or TRAINING['image_size']
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=AUGMENTATION['normalize_mean'],
            std=AUGMENTATION['normalize_std'],
        ),
    ])


def load_dataset_from_folders(root_dir):
    """
    Load dataset organized as root/class_name/image.jpg.

    Returns:
        image_paths: list of file paths
        labels: list of integer labels
        class_names: sorted list of class names
    """
    image_paths = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    for idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls_name)
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'):
            for img_path in glob.glob(os.path.join(cls_dir, ext)):
                image_paths.append(img_path)
                labels.append(idx)

    return image_paths, labels, class_names


def find_dataset_root(dataset_name):
    """Find the actual image root directory for a dataset."""
    base_dir = DATASET_PATHS[dataset_name]

    if dataset_name == 'eurosat':
        # EuroSAT via torchvision extracts to eurosat/eurosat/2750/
        candidates = [
            base_dir,
            os.path.join(base_dir, 'EuroSAT'),
            os.path.join(base_dir, '2750'),
            os.path.join(base_dir, 'EuroSAT_RGB'),
            os.path.join(base_dir, 'eurosat', '2750'),
        ]
    elif dataset_name == 'ucmerced':
        candidates = [
            base_dir,
            os.path.join(base_dir, 'UCMerced_LandUse', 'Images'),
            os.path.join(base_dir, 'Images'),
            os.path.join(base_dir, 'UCMerced_LandUse'),
        ]
    elif dataset_name == 'nwpu_resisc45':
        candidates = [
            base_dir,
            os.path.join(base_dir, 'NWPU-RESISC45'),
        ]
    elif dataset_name == 'aid':
        candidates = [
            base_dir,
            os.path.join(base_dir, 'AID'),
        ]
    else:
        candidates = [base_dir]

    for candidate in candidates:
        if os.path.isdir(candidate):
            # Check if this dir contains class subdirectories with images
            subdirs = [d for d in os.listdir(candidate)
                       if os.path.isdir(os.path.join(candidate, d))]
            if len(subdirs) >= 5:  # At least some class directories
                return candidate

    return base_dir


def create_dataloaders(dataset_name, train_ratio=None, batch_size=None,
                       image_size=None, seed=42, verbose=True):
    """
    Create train/test DataLoaders for a dataset.

    Args:
        dataset_name: 'eurosat', 'nwpu_resisc45', or 'aid'
        train_ratio: fraction for training (default from config)
        batch_size: batch size (default from config)
        image_size: resize target (default 224)
        seed: random seed
        verbose: print info

    Returns:
        train_loader, test_loader, class_names
    """
    if train_ratio is None:
        train_ratio = DATASETS[dataset_name]['train_ratio']
    if batch_size is None:
        batch_size = TRAINING['batch_size']
    if image_size is None:
        image_size = TRAINING['image_size']

    root_dir = find_dataset_root(dataset_name)
    image_paths, labels, class_names = load_dataset_from_folders(root_dir)

    if verbose:
        print(f"Dataset: {dataset_name}")
        print(f"  Root: {root_dir}")
        print(f"  Images: {len(image_paths)}")
        print(f"  Classes: {len(class_names)}")

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio,
                                 random_state=seed)
    train_idx, test_idx = next(sss.split(image_paths, labels))

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    if verbose:
        print(f"  Train: {len(train_paths)}, Test: {len(test_paths)}")

    train_dataset = SceneDataset(
        train_paths, train_labels,
        transform=get_train_transform(image_size)
    )
    test_dataset = SceneDataset(
        test_paths, test_labels,
        transform=get_test_transform(image_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=min(TRAINING['num_workers'], 2), pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=min(TRAINING['num_workers'], 2), pin_memory=True
    )

    return train_loader, test_loader, class_names


def download_eurosat(target_dir=None):
    """Download EuroSAT RGB dataset."""
    import urllib.request

    if target_dir is None:
        target_dir = DATASET_PATHS['eurosat']
    os.makedirs(target_dir, exist_ok=True)

    url = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip'
    zip_path = os.path.join(target_dir, 'EuroSAT_RGB.zip')

    if not os.path.exists(zip_path):
        print(f"Downloading EuroSAT from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Saved to {zip_path}")

    # Extract
    print("Extracting EuroSAT...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)
    print("  Done.")


def download_ucmerced(target_dir=None):
    """Download UC Merced Land Use dataset (2,100 images, 21 classes)."""
    import urllib.request

    if target_dir is None:
        target_dir = DATASET_PATHS['ucmerced']
    os.makedirs(target_dir, exist_ok=True)

    url = 'http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip'
    zip_path = os.path.join(target_dir, 'UCMerced_LandUse.zip')

    images_dir = os.path.join(target_dir, 'UCMerced_LandUse', 'Images')
    if os.path.isdir(images_dir):
        subdirs = [d for d in os.listdir(images_dir)
                   if os.path.isdir(os.path.join(images_dir, d))]
        if len(subdirs) >= 21:
            print(f"UC Merced already downloaded at {images_dir}")
            return

    if not os.path.exists(zip_path):
        print(f"Downloading UC Merced from {url}...")
        print("  (~317 MB, may take a few minutes)")
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Saved to {zip_path}")

    print("Extracting UC Merced...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)
    print("  Done.")


def download_nwpu_resisc45(target_dir=None):
    """
    NWPU-RESISC45 requires manual download from OneDrive.
    Print instructions for the user.
    """
    if target_dir is None:
        target_dir = DATASET_PATHS['nwpu_resisc45']
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print("NWPU-RESISC45 MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print()
    print("1. Go to: https://gcheng-nwpu.github.io/#Datasets")
    print("   OR OneDrive link from the paper")
    print("2. Download NWPU-RESISC45.rar (~400 MB)")
    print(f"3. Extract to: {target_dir}/NWPU-RESISC45/")
    print("   Structure: NWPU-RESISC45/airplane/airplane_001.jpg ...")
    print()


def download_aid(target_dir=None):
    """
    AID requires manual download from Google Drive or Baidu.
    Print instructions for the user.
    """
    if target_dir is None:
        target_dir = DATASET_PATHS['aid']
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print("AID MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print()
    print("1. Go to: https://captain-whu.github.io/AID/")
    print("2. Download AID.zip (~2.4 GB)")
    print(f"3. Extract to: {target_dir}/AID/")
    print("   Structure: AID/Airport/airport_00001.jpg ...")
    print()


def verify_dataset(dataset_name, verbose=True):
    """Check if a dataset is available and properly structured."""
    root_dir = find_dataset_root(dataset_name)
    expected_classes = DATASETS[dataset_name]['num_classes']

    if not os.path.isdir(root_dir):
        if verbose:
            print(f"  {dataset_name}: NOT FOUND at {root_dir}")
        return False, 0, 0

    image_paths, labels, class_names = load_dataset_from_folders(root_dir)

    ok = len(class_names) == expected_classes and len(image_paths) > 0
    if verbose:
        status = "OK" if ok else "INCOMPLETE"
        print(f"  {dataset_name}: {status} - {len(class_names)} classes, "
              f"{len(image_paths)} images")

    return ok, len(class_names), len(image_paths)
