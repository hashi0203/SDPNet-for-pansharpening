dr = 1050.0
dr_test = 20000.0
# dr_test = 23600.0
off_test = 500

# model_date = "1203"
# MS2P = "4800"
# P2MS = "4700"
# SPAT = "4900"
# SPEC = "4600"

model_date = "1205"
MS2P = "9800"
P2MS = "9800"
SPAT = "3900"
SPEC = "8700"

MS2P_MODEL_SAVEPATH = './MS2P_models/%s/%s/%s.ckpt' % (model_date, MS2P, MS2P)
P2MS_MODEL_SAVEPATH = './P2MS_models/%s/%s/%s.ckpt' % (model_date, P2MS, P2MS)
SPAT_MODEL_SAVEPATH = './spat_models/%s/%s/%s.ckpt' % (model_date, SPAT, SPAT)
SPEC_MODEL_SAVEPATH = './spec_models/%s/%s/%s.ckpt' % (model_date, SPEC, SPEC)

SPAT_DIFF_SAVEPATH = './spat_diffs/spat_diff_%s.txt' % model_date
SPEC_DIFF_SAVEPATH = './spec_diffs/spec_diff_%s.txt' % model_date

MODEL = 12280
MODEL_SAVE_PATH = './models/%s/%s/%s.ckpt' % (model_date, MODEL, MODEL)
OUTPUT_PATH = './results/%s/' % model_date
