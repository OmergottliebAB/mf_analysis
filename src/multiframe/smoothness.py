import logging
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger("mf_analyser")

def second_derivative_anomaly(x, age):
    # TODO: add sign change between consecutive anomalies
    def anomaly(value, avg, std, age, index):
        three_std_cond = (value > avg+3*std) or (value < avg-3*std)
        age_cond = age[index] >= 10 and age[index] < age[-1]
        unconfirmed_cond = age[index]-age[index-2] == 2
        return three_std_cond and age_cond and unconfirmed_cond
    
    second_derivative = np.diff(x, n=2)
    avg, std = np.mean(second_derivative), np.std(second_derivative)
    # find indices of anomalies in second derivative and their age is larger than 10
    indices = [index for index, value in enumerate(second_derivative) if anomaly(value, avg, std, age, index+2)]
    if indices:
        return True, indices
    else:
        return False, []
    

