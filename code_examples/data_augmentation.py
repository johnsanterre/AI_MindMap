"""Data Augmentation — image transforms for training data expansion."""

import numpy as np

def flip_horizontal(img):
    return img[:, ::-1]

def flip_vertical(img):
    return img[::-1, :]

def rotate_90(img, k=1):
    return np.rot90(img, k)

def add_noise(img, std=0.1):
    return np.clip(img + np.random.randn(*img.shape) * std, 0, 1)

def random_crop(img, crop_h, crop_w):
    h, w = img.shape[:2]
    y = np.random.randint(0, h - crop_h + 1)
    x = np.random.randint(0, w - crop_w + 1)
    return img[y:y+crop_h, x:x+crop_w]

def cutout(img, size=4):
    """Random erasing / cutout augmentation."""
    h, w = img.shape[:2]
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    y1, y2 = max(0, y-size//2), min(h, y+size//2)
    x1, x2 = max(0, x-size//2), min(w, x+size//2)
    aug = img.copy()
    aug[y1:y2, x1:x2] = 0
    return aug

def mixup(img1, img2, label1, label2, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1-lam) * img2
    mixed_label = lam * label1 + (1-lam) * label2
    return mixed_img, mixed_label

# --- demo ---
np.random.seed(42)
# create a simple 8x8 "image"
img = np.zeros((8, 8))
img[2:6, 2:6] = 1.0  # white square in center
img[3:5, 3:5] = 0.5  # gray inner

print("Original 8x8 image:")
print((img * 9).astype(int))

augmentations = [
    ("Flip H", flip_horizontal(img)),
    ("Flip V", flip_vertical(img)),
    ("Rot 90", rotate_90(img)),
    ("Noise", add_noise(img, std=0.2)),
    ("Cutout", cutout(img, size=3)),
]

for name, aug in augmentations:
    print(f"\n{name}:")
    print((aug * 9).astype(int).clip(0, 9))

# mixup demo
img2 = np.eye(8)
mixed, label = mixup(img, img2, 0, 1, alpha=0.4)
print(f"\nMixup (λ={label:.2f}):")
print((mixed * 9).astype(int))
