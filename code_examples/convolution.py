"""Convolution — 2D convolution from scratch."""

import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """2D convolution of a single-channel image with a kernel."""
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(patch * kernel)
    return output

# --- demo ---
np.random.seed(42)
image = np.random.randn(8, 8).round(1)

# common kernels
edge_detect = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

blur = np.ones((3, 3)) / 9

sharpen = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]])

print("Input (8x8):")
print(image.round(1))

for name, kernel in [("Edge detect", edge_detect), ("Blur", blur), ("Sharpen", sharpen)]:
    result = conv2d(image, kernel, padding=1)
    print(f"\n{name} output (8x8):")
    print(result.round(1))
