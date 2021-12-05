import numpy as np
import cv2
from path import Path
import rasterio
import matplotlib.pyplot as plt
from scipy.misc import imread
import config

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
train_pans = []
train_mss = []

for data_name in data_names:
    print("loading %s.." % data_name)
    in_pan_path = (data_dir / data_name).files("*-PAN*.tiff")[0]
    in_ms_path = (data_dir / data_name).files("*-MSS*.tiff")[0]

    train_pans.append(downsample_pan(rasterio.open(in_pan_path).read(1)) / config.dr)
    train_mss.append(np.stack([rasterio.open(in_ms_path).read(c+1) for c in range(ms_meta['count'])], axis=2) / config.dr)

test_pan = (imread('test_imgs/pan_org/2.png') + config.off_test) / config.dr_test
test_ms = (np.stack([rasterio.open('test_imgs/ms_org/2.tif').read(c+1) for c in range(4)], axis=2) + config.off_test) / config.dr_test

def hist(imgs, filename):
    imgs = np.array(imgs).reshape(-1)
    plt.figure()
    weights = np.ones(len(imgs)) / len(imgs)
    plt.hist(imgs, weights=weights)
    plt.title('min: %.2f, max: %.2f, ave: %.2f, std: %.2f' % (np.min(imgs), np.max(imgs), np.mean(imgs), np.std(imgs)))
    plt.savefig(filename)

hist(train_pans, "train_pans.png")
hist(train_mss, "train_mss.png")
hist(test_pan, "test_pan.png")
hist(test_ms, "test_ms.png")
