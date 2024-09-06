import numpy as np
from numba import jit

@jit(nopython=True)
def hybrid_median_filter(img, filter_size):
    result = np.zeros_like(img)
    height, width, channel = img.shape
    pad = filter_size // 2 + 1
    img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    # =====================================
    index = 0
    count = 0
    # =====================================

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            for k in range(channel):

                # =====================================
                if (count == 10000):
                    percent = index / (height * width * 3) * 100
                    print(percent, "%")
                    count = 0
                index += 1
                count += 1
                # =====================================

                filter = []
                filter_1 = []
                filter_2 = []
                filter_3 = []
                filter_4 = []
                filter_5 = []
                filter_6 = []
                filter_7 = []
                filter_8 = []

                for p in range(1, pad):
                    filter_1.append(img_pad[i + p, j, k])
                    filter_2.append(img_pad[i - p, j, k])
                    filter_3.append(img_pad[i, j + p, k])
                    filter_4.append(img_pad[i, j - p, k])
                    filter_5.append(img_pad[i + p, j + p, k])
                    filter_6.append(img_pad[i - p, j - p, k])
                    filter_7.append(img_pad[i - p, j + p, k])
                    filter_8.append(img_pad[i + p, j - p, k])

                filter.append(img_pad[i,j,k])
                filter.append(np.median(filter_1))
                filter.append(np.median(filter_2))
                filter.append(np.median(filter_3))
                filter.append(np.median(filter_4))
                filter.append(np.median(filter_5))
                filter.append(np.median(filter_6))
                filter.append(np.median(filter_7))
                filter.append(np.median(filter_8))

                result[i-pad, j-pad, k] = np.median(filter)

    return result