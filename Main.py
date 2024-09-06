import io, os
from filters.gaussian import gaussian_filter_opencv
from filters.hybrid_median import hybrid_median_filter
from filters.mean import mean_filter_opencv
from filters.median import median_filter
from filters.non_local_mean import non_local_mean_filter
from skimage import metrics

def calculate_psnr(original, noisy):
    return metrics.peak_signal_noise_ratio(original, noisy)

def load_images(path):
    images = []
    image_names = []

    for filename in os.listdir(path):
        if filename.endswith('.png'):
            image_names.append(filename)
    
    image_names.sort()

    for filename in image_names:
        img_path = os.path.join(path, filename)
        img = io.imread(img_path)
        if img is not None:
            images.append(img)

    return images, image_names


img = io.imread("image/set1/noisy/dog_noisy.png")

for windowSize in range(5, 30, 5):
    for patchSize in range(2, 12, 2):
        if windowSize <= patchSize:
            continue
        for sigma in range(25, 50, 5):
            result = non_local_mean_filter(img, windowSize, patchSize, sigma)
            io.imsave("image/set1/result/dog_nlm_"+str(windowSize)+"_"+str(patchSize)+"_"+str(sigma)+".png", result)

img, imgName = load_images("image/set2/result/median/")
for index, name in enumerate(imgName):
    for windowSize in range(15, 20, 5):
        for patchSize in range(2, 4, 2):
            if windowSize <= patchSize:
                continue
            for sigma in range(10, 15, 5):
                result = non_local_mean_filter(img[index], windowSize, patchSize, sigma)
                io.imsave("image/set1/result/"+name.replace(".png", "_")+str(windowSize)+"_"+str(patchSize)+"_"+str(sigma)+".png", result)

img = io.imread("image/set2/noisy/card_noisy.png")

result = non_local_mean_filter(img, 15, 2, 10)
io.imsave("image/set2/result/nlm/card_nlm_15_2_10.png", result)