import cv2
import numpy as np


def Butterworth(sp, args):
    P = sp[0] / 2
    Q = sp[1] / 2
    U, V = np.meshgrid(range(sp[0]), range(
        sp[1]), sparse=False, indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    H = 1/(1+(Duv/args[0]**2)**args[1])
    return 1-H


def Gaussian(sp, args):
    P = sp[0] / 2
    Q = sp[1] / 2
    U, V = np.meshgrid(range(sp[0]), range(
        sp[1]), sparse=False, indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    H = np.exp((-Duv/(2*args[0]**2)))
    return 1-H


def applyFilter(I, H, a, b):
    H = np.fft.fftshift(H)
    return (a+b*H)*I


def homomorphicFilt(img, args, a, b, type='Gaussian', ):
    imgLog = np.log1p(np.array(img, dtype="float"))
    imgfft = np.fft.fft2(imgLog)
    if type == 'Butterworth':
        H = Butterworth(imgfft.shape, args)
    elif type == 'Gaussian':
        H = Gaussian(imgfft.shape, args)
    else:
        raise(Exception(f'undefined type {type}'))
    imgfftTrans = applyFilter(imgfft, H, a, b)
    imgifft = np.fft.ifft2(imgfftTrans)
    img = np.exp(np.real(imgifft)) - 1
    return np.uint8(img)


def main():
    path = 'H:\\Learning\DIP\\HomomorphicFilter\\test\\in2.jpg'
    outpath = 'H:\\Learning\DIP\\HomomorphicFilter\\test\\out2.jpg'

    img = cv2.imread(path)
    a = 0.90
    b = 1.5
    sig = [350, 10]  # 降低sig0粒度变大
    type = 'Gaussian'
    b = homomorphicFilt(img[:, :, 0], sig, a, b, type)
    g = homomorphicFilt(img[:, :, 1], sig, a, b, type)
    r = homomorphicFilt(img[:, :, 2], sig, a, b, type)
    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i][j][0] = b[i][j]
            res[i][j][1] = g[i][j]
            res[i][j][2] = r[i][j]
    cv2.imwrite(outpath, res)


if __name__ == '__main__':
    main()
