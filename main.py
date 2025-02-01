import numpy as np

with np.load("polar/train/0.npz") as data:
    print(data["images"].shape)