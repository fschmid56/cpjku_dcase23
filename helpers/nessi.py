# Complexity Calculator imported from :
# https://github.com/AlbertoAncilotto/NeSsi/blob/main/nessi.py

import torchinfo


def get_model_size(model, input_size):
    model_profile = torchinfo.summary(
        model,
        input_size=input_size
    )
    return model_profile.total_mult_adds, model_profile.total_params
