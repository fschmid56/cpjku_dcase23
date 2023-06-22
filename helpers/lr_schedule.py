import numpy as np


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    """
    Simple learning rate scheduler. This function returns the factor the maximum
     learning rate is multiplied with. It includes:
    1. Warmup Phase: lr exponentially increases for 'warmup' number of epochs (to a factor of 1.0)
    2. Constant LR Phase: lr reaches max value (factor of 1.0)
    3. Linear Decrease Phase: lr decreases linearly starting from epoch 'start_rampdown'
    4. Finetuning Phase: phase 3 completes after 'rampdown_length' epochs, followed by a finetuning phase using
                        a learning rate of max lr * 'last_value'
    """
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)
    def wrapper(epoch):
        return rampup(epoch) * rampdown(epoch)
    return wrapper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def wrapper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    def wrapper(epoch):
        if epoch <= start:
            return 1.
        elif epoch - start < rampdown_length:
            return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value
    return wrapper
