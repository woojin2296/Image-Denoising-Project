import cv2

def mean_filter_opencv(img, filter_size):
	kernel_size = filter_size // 2 + 1
	result = cv2.blur(img, (kernel_size, kernel_size))
	return result