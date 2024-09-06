import cv2
import numpy as np

def median_filter_opencv(img, filter_size):
	return cv2.medianBlur(img, filter_size)

def median_filter(img, filter_size):
    height, width, chan = img.shape
    result = np.zeros_like(img)
    pad = filter_size // 2
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            for c in range(chan):
                result[i-pad,j-pad,c] = np.median(
                    img_padded[i-pad:i+pad+1,j-pad:j+pad+1,c]
                )
    return result