# dr = 1050.0
# dr_test = 23600.0
# off_test = 0
# # dr_test = 20000.0
# # off_test = 500

# pan_path = 'PAN.h5'
# gt_path = 'GT.h5'
# pan_test_imgs_path = 'test_imgs/pan/'
# ms_test_imgs_path = 'test_imgs/ms/'

# model_date = "1203"
# MS2P = "4800"
# P2MS = "4700"
# SPAT = "4900"
# SPEC = "4600"
# MODEL = "12280"

# model_date = "1205"
# MS2P = "9800"
# P2MS = "9800"
# SPAT = "3900"
# SPEC = "8700"
# MODEL = "13200"

# model_date = "1205-2"
# MS2P = "4000"
# P2MS = "9200"
# SPAT = "9300"
# SPEC = "12275"
# MODEL = "28300"

dr = 23600.0
dr_test = 23600.0
off_test = 0

pan_path = 'PAN-test.h5'
gt_path = 'GT-test.h5'
pan_test_imgs_path = 'test_imgs/pan_test/'
ms_test_imgs_path = 'test_imgs/ms_test/'

# model_date = "1205-3"
# MS2P = "4500"
# P2MS = "4300"
# SPAT = "3500"
# SPEC = "5500"
# MODEL = "7600"

model_date = "1205-4"
MS2P = "8100"
P2MS = "7700"
SPAT = "7700"
SPEC = "7600"
MODEL = "0"

MS2P_MODEL_SAVEPATH = './MS2P_models/%s/%s/%s.ckpt' % (model_date, MS2P, MS2P)
P2MS_MODEL_SAVEPATH = './P2MS_models/%s/%s/%s.ckpt' % (model_date, P2MS, P2MS)
SPAT_MODEL_SAVEPATH = './spat_models/%s/%s/%s.ckpt' % (model_date, SPAT, SPAT)
SPEC_MODEL_SAVEPATH = './spec_models/%s/%s/%s.ckpt' % (model_date, SPEC, SPEC)

SPAT_DIFF_SAVEPATH = './spat_diffs/spat_diff_%s.txt' % model_date
SPEC_DIFF_SAVEPATH = './spec_diffs/spec_diff_%s.txt' % model_date

MODEL_SAVE_PATH = './models/%s/%s/%s.ckpt' % (model_date, MODEL, MODEL)
OUTPUT_PATH = './results/%s/%d-%d/' % (model_date, dr_test, off_test)
