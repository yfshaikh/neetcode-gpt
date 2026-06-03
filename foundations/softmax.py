import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        denom = 0
        max_val = max(z)
        for i in range(len(z)):
            denom += math.exp(z[i]-max_val)
        for i in range(len(z)):
            num = math.exp(z[i]-max_val)
            z[i] = round(num / denom, 4)
        return z
