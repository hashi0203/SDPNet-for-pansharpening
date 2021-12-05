import os
import config
model_date = config.model_date

os.mkdir('./MS2P_models/%s' % model_date)
os.mkdir('./P2MS_models/%s' % model_date)
os.mkdir('./spat_models/%s' % model_date)
os.mkdir('./spec_models/%s' % model_date)
os.mkdir('./models/%s' % model_date)
