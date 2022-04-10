import cv2
import numpy as np
import copy
import math

prewitt_x = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]
)

prewitt_y = prewitt_x.T

sobel_x = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
)

sobel_y = sobel_x.T

laplace = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]
)

simpleLaplace = np.array(
    [
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ]
)


def averageGray(rgb):
    return (np.dot(rgb[..., :3], [1, 1, 1])/3)


def averageGrayWithWeighted(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def convolution(sourceImage, operator, size=3):
    image = copy.deepcopy(sourceImage)
    imageWidth = image.shape[0]
    imageHeight = image.shape[1]
    newImage = np.zeros([imageWidth - size, imageHeight - size])
    for width in range(imageWidth - size):
        for height in range(imageHeight - size):
            newImage[width, height] = np.sum(
                sourceImage[width:width + size, height:height + size] * operator)
    return newImage.astype(np.uint8)


def getGaussianMarix(size=3, padding=1):
    sigma = size / 3.0
    gaussian = np.zeros([size, size])
    sum = 0
    for x in range(size):
        for y in range(size):
            gaussian[x, y] = math.exp(-1/2 * (np.square(x-padding)/np.square(sigma) + (
                np.square(y-padding)/np.square(sigma)))) / (2*math.pi*sigma*sigma)
            sum = sum + gaussian[x, y]
    gaussian = gaussian / sum
    return gaussian


def cannyKernel(sourceImage):
    image = copy.deepcopy(sourceImage)
    imageWidth, imageHeight = image.shape
    imageDX = np.zeros([imageWidth - 1, imageHeight - 1])
    imageDY = np.zeros([imageWidth - 1, imageHeight - 1])
    imageD = np.zeros([imageWidth - 1, imageHeight - 1])
    image = image.astype(np.float64)
    for w in range(imageWidth - 1):
        for h in range(imageHeight - 1):
            imageDX[w, h] = (image[w, h + 1] - image[w, h] +
                             image[w + 1, h + 1] - image[w + 1, h]) / 2
            imageDY[w, h] = (image[w + 1, h] - image[w, h] +
                             image[w + 1, h] - image[w + 1, h + 1]) / 2
            imageD[w, h] = np.sqrt(
                np.square(imageDX[w, h]) + np.square(imageDY[w, h]))
    newImage = copy.deepcopy(imageD)
    nImageWidth, nImageHeight = imageD.shape
    newImage[0, :] = newImage[:, 0] = newImage[nImageWidth -
                                               1, :] = newImage[:, nImageHeight-1] = 0
    for w in range(1, nImageWidth - 1):
        for h in range(1, nImageHeight - 1):
            if (imageD[w, h] == 0).all():
                newImage[w, h] = 0
            else:
                gradX = imageDX[w, h]
                gradY = imageDY[w, h]
                grad = imageD[w, h]
                if(np.abs(gradX) > np.abs(gradY)).all():
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = imageD[w, h - 1]
                    grad4 = imageD[w, h + 1]
                    if (gradX * gradY > 0).all():
                        grad1 = imageD[w + 1, h - 1]
                        grad3 = imageD[w - 1, h + 1]
                    else:
                        grad1 = imageD[w - 1, h - 1]
                        grad3 = imageD[w + 1, h + 1]
                else:
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = imageD[w - 1, h]
                    grad4 = imageD[w + 1, h]
                    if (gradX * gradY > 0).all():
                        grad1 = imageD[w - 1, h - 1]
                        grad3 = imageD[w + 1, h + 1]
                    else:
                        grad1 = imageD[w - 1, h + 1]
                        grad3 = imageD[w + 1, h - 1]
                gradTmp1 = weight * grad1 + (1 - weight) * grad2
                gradTmp2 = weight * grad3 + (1 - weight) * grad4
                if (grad >= gradTmp1).all() and (grad >= gradTmp2).all():
                    newImage[w, h] = grad
                else:
                    newImage[w, h] = 0
    return newImage.astype(np.uint8)


def cannyFinal(sourceImage):
    image = copy.deepcopy(sourceImage)
    imageWidth, imageHeight = image.shape
    num1 = 0.1 * sourceImage.max()
    num2 = 0.2 * sourceImage.max()
    for w in range(1, imageWidth - 1):
        for h in range(1, imageHeight - 1):
            if (sourceImage[w, h] < num1).all():
                image[w, h] = 0
            elif (sourceImage[w, h] > num2).all():
                image[w, h] = 255
            elif ((sourceImage[w - 1, h - 1:h + 1] < num1).any() or (sourceImage[w + 1, h - 1:h + 1]).any() or (sourceImage[w, [h - 1, h + 1]] < num2).any()):
                image[w, h] = 255
    return image.astype(np.uint8)


def main():
    path = 'H:\\Learning\DIP\\EdgeDetection\\test\\in2.jpg'
    outpath = 'H:\\Learning\DIP\\EdgeDetection\\test\\canny4.jpg'
    img = cv2.imread(path)
    # cv2.imwrite(outpath, cannyFinal(cannyKernel(averageGray(img))))
    cv2.imwrite(outpath, cannyFinal(cannyKernel(
        convolution(averageGray(img), getGaussianMarix()))))
    # cv2.imwrite(outpath, convolution(averageGray(img), getGaussianMarix()))


if __name__ == '__main__':
    main()
