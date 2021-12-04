import numpy as np
import cv2
from path import Path
import random
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
pan_path = Path("test_imgs/pan")
ms_path = Path("test_imgs/ms")

pan_data = []
ms_data = []
for data_name in data_names:
    print("loading %s.." % data_name)
    in_pan_path = (data_dir / data_name).files("*-PAN*.tiff")[0]
    in_ms_path = (data_dir / data_name).files("*-MSS*.tiff")[0]

    pan_data.append(rasterio.open(in_pan_path).read(1))
    ms_data.append(np.stack([rasterio.open(in_ms_path).read(c+1) for c in range(ms_meta['count'])], axis=2))

pan_patch_size = 264
ms_patch_size = pan_patch_size // 4
max_patch_h = min(pan_data[0].shape[0], ms_data[0].shape[0] * 4) // 4 - ms_patch_size
max_patch_w = min(pan_data[0].shape[1], ms_data[0].shape[1] * 4) // 4 - ms_patch_size

pan_meta_out = pan_meta
pan_meta_out['width'] = pan_meta_out['height'] = pan_patch_size
ms_meta_out = ms_meta
ms_meta_out['width'] = ms_meta_out['height'] = ms_patch_size

for i in range(100):
    data_idx = random.randint(0, len(data_names)-1)
    h = random.randint(0, max_patch_h-1)
    w = random.randint(0, max_patch_w-1)

    pan = pan_data[data_idx][h*4:h*4+pan_patch_size, w*4:w*4+pan_patch_size]
    ms = ms_data[data_idx][h:h+ms_patch_size, w:w+ms_patch_size]

    with rasterio.open(pan_path / ('%d.tif' % (i+1)), 'w', **pan_meta_out) as dst:
        dst.write(pan, 1)

    with rasterio.open(ms_path / ('%d.tif' % (i+1)), 'w', **ms_meta_out) as dst:
        for j in range(ms_meta_out['count']):
            dst.write(ms[:,:,j], j+1)
