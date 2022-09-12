import cv2
import numpy as np
from skimage.util import random_noise

img = cv2.imread("lena.png")
cv2.imshow('original image',img)

gray=cv2.imread("lena.png",0)
cv2.imshow("Grayscale image",gray)

def salt_and_pepper_noise(times, f):
	i1=cv2.imread("lena.png")
	s=0
	for i in range(times):
		noise_img = random_noise(i1, mode='s&p',amount=0.1)
		s=s+noise_img
	final_img = np.array(255*s/times, dtype = 'uint8')
	cv2.imwrite(f, final_img)
	
def gaussian_noise(times, g):
	i2=cv2.imread("lena.png")
	s1=0
	for i in range(times):
		noise_img = random_noise(i2, mode='gaussian', mean=0, var=0.01)
		s1=s1+noise_img
	final_img = np.array(255*s1/times, dtype = 'uint8')
	cv2.imwrite(g, final_img)
	
def speckle_noise(times, h):
	i3=cv2.imread("lena.png")
	s2=0
	for i in range(times):
		noise_img = random_noise(i3, mode='speckle', mean=0.3, var=0.01)
		s2=s2+noise_img
	final_img = np.array(255*s2/times, dtype = 'uint8')
	cv2.imwrite(h, final_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
salt_and_pepper_noise(5, "sp_image_5.png")
salt_and_pepper_noise(10, "sp_image_10.png")
salt_and_pepper_noise(15, "sp_image_15.png")
salt_and_pepper_noise(20, "sp_image_20.png")
salt_and_pepper_noise(25, "sp_image_25.png")
salt_and_pepper_noise(30, "sp_image_30.png")

gaussian_noise(5, "g_image_5.png")
gaussian_noise(10, "g_image_10.png")
gaussian_noise(15, "g_image_15.png")
gaussian_noise(20, "g_image_20.png")
gaussian_noise(25, "g_image_25.png")
gaussian_noise(30, "g_image_30.png")

speckle_noise(5, "s_image_5.png")
speckle_noise(10, "s_image_10.png")
speckle_noise(15, "s_image_15.png")
speckle_noise(20, "s_image_20.png")
speckle_noise(25, "s_image_25.png")
speckle_noise(30, "s_image_30.png")
