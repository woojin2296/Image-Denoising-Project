import cv2

def gaussian_filter_opencv(image, kernel_size, sigma):
    result = cv2.GaussianBlur(image, kernel_size, sigma)
    return result