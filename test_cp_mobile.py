from models.cp_mobile_clean import get_model
from helpers.nessi import get_model_size

cp_mobile = get_model(base_channels=16,
                      channels_multiplier=1.5,
                      expansion_rate=1.75)

cp_mobile.eval().fuse_model()

get_model_size(cp_mobile, (1, 1, 256, 64))
