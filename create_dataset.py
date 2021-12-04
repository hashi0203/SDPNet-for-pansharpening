import numpy as np
import cv2
from path import Path
import h5py
import rasterio

data_dir = Path("/home/share/dataset/GF-2/2017")
data_names = [Path("GF2_PMS1_E100.5_N37.2_20171013_L1A0002678101"),
              Path("GF2_PMS1_E100.5_N37.4_20171013_L1A0002678097"),
              Path("GF2_PMS1_E100.6_N37.6_20171013_L1A0002678096"),
              Path("GF2_PMS2_E100.3_N37.4_20170810_L1A0002534662"),
              Path("GF2_PMS2_E100.5_N36.7_20170805_L1A0002526723"),
              Path("GF2_PMS2_E100.7_N37.2_20171013_L1A0002672923"),
              Path("GF2_PMS2_E100.7_N37.4_20171013_L1A0002672921")]

pan_meta = rasterio.open((data_dir / data_names[0]).files("*-PAN*.tiff")[0]).profile
ms_meta = rasterio.open((data_dir / data_names[0]).files("*-MSS*.tiff")[0]).profile

pan_path = 'PAN.h5'
gt_path = 'GT.h5'

def downsample_pan(img):
    height, width = img.shape
    return cv2.resize(img, (width // 4, height // 4))

patch_size = 264
pan_patches = []
ms_patches = []

for data_name in data_names:
    print("loading %s.." % data_name)
    in_pan_path = (data_dir / data_name).files("*-PAN*.tiff")[0]
    in_ms_path = (data_dir / data_name).files("*-MSS*.tiff")[0]

    pan = downsample_pan(rasterio.open(in_pan_path).read(1))
    ms = np.stack([rasterio.open(in_ms_path).read(c+1) for c in range(ms_meta['count'])], axis=2)

    n_patch_h = min(pan.shape[0], ms.shape[0]) // patch_size
    n_patch_w = min(pan.shape[1], ms.shape[1]) // patch_size

    for i in range(n_patch_h):
        for j in range(n_patch_w):
            pan_patches.append(pan[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])
            ms_patches.append(ms[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])

pan_patches = np.array(pan_patches).reshape(-1, 1, patch_size, patch_size)
ms_patches = np.array(ms_patches).transpose(0, 3, 1, 2)

print("number of data: %d" % pan_patches.shape[0])

with h5py.File(pan_path, 'w') as f:
    f.create_dataset('data', data=pan_patches)

with h5py.File(gt_path, 'w') as f:
    f.create_dataset('data', data=ms_patches)
