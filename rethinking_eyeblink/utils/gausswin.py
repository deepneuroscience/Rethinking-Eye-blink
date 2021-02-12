# Rethinking Eye-blink project

## Author(s): Dr. Youngjun Cho *(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com

import numpy as np

#Gaussian window function
def gausswin(L, alpha=2.5):
    N = L - 1
    n = np.arange(0,N+1)-N/2
    w = np.exp(-(1/2)*(alpha*n /(N /2))**2)
#     return w[0:-1]
    return w