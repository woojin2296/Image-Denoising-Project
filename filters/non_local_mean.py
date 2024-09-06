import numpy as np
from numba import jit

@jit(nopython=True)
def non_local_mean_filter(img, windowSize, patchSize, sigma, verbose=True):
    height, width, chan = img.shape

    padwidth = windowSize // 2
    patchRad = patchSize // 2

    paddedImage = np.zeros((height + windowSize, width + windowSize, chan), dtype=np.uint8)
    paddedImage[padwidth:padwidth + height, padwidth:padwidth + width, :] = img

    outputImage = paddedImage.copy()

    if verbose:
        iterator = 0
        totalIterations = height * width * (windowSize - patchSize) ** 2
        print("TOTAL ITERATIONS = ", totalIterations)

    for h in range(padwidth, padwidth + height):
        for w in range(padwidth, padwidth + width):
            winw = w - padwidth
            winh = h - padwidth

            pixelColor = np.zeros(chan)
            totalWeight = 0

            originalPatch = paddedImage[h - patchRad:h + patchRad + 1, w - patchRad:w + patchRad + 1, :]

            for patchh in range(winh, winh + windowSize - patchSize):
                for patchw in range(winw, winw + windowSize - patchSize):
                    compPatch = paddedImage[patchh:patchh + patchSize + 1, patchw:patchw + patchSize + 1, :]

                    euclideanDistance = np.sqrt(np.sum((compPatch - originalPatch) ** 2))
                    weight = np.exp(-euclideanDistance / (sigma**2) / 2)
                    totalWeight += weight

                    pixelColor += weight * paddedImage[patchh + patchRad, patchw + patchRad, :]
                    iterator += 1

                    if verbose and iterator % 1000000 == 0:
                        percentComplete = iterator * 100 / totalIterations
                        print('COMPLETE =', round(percentComplete, 5), "%")

            pixelColor /= totalWeight
            outputImage[h, w, :] = pixelColor

    return outputImage[padwidth:padwidth + height, padwidth:padwidth + width, :]