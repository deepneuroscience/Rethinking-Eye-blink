# Rethinking Eye-blink project

## Author(s): Dr. Youngjun Cho *(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com


import numpy as np
import math

def cal_ratio_size(t, b, l, r):
    w = math.hypot((l[0] - r[0]), (l[1] - r[1]))
    h = math.hypot((t[0] - b[0]), (t[1] - b[1]))

    if h != 0:
        ratio = w / h
    else:
        ratio = np.nan
    size = h * w
    return (ratio, size)