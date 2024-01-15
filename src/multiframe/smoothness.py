import logging
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger("mf_analyser")

def second_derivative_anomaly(x, age):
    # TODO: add last frame condition
    # TODO: add sign change between consecutive anomalies
    # TODO: unconfirmed frames logic
    second_derivative = np.diff(x, n=2)
    avg, std = np.mean(second_derivative), np.std(second_derivative)
    # find indices of anomalies in second derivative and their age is larger than 10
    indices = [index for index, value in enumerate(second_derivative) if (((value > avg+3*std) or (value < avg-3*std)) and age[index] >= 10)]
    if indices:
        return True, indices
    else:
        return False, []
    

