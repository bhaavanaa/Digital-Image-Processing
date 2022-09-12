import cv2
import numpy as np
from skimage.util import random_noise


img = cv2.imread("lena.png")
cv2.imshow('original image',img)


gray=cv2.imread("lena.png",0)
cv2.imwrite("Grayscale_image.png",gray)


def salt_and_pepper_noise(times, f, k):
	i1=cv2.imread("Grayscale_image.png")
	s=0
	for i in range(times):
		noise_img = random_noise(i1, mode='s&p',amount=k)
		s=s+noise_img
	final_img = np.array(255*s/times, dtype = 'uint8')
	cv2.imwrite(f, final_img)


def med_filter(name):
	s=cv2.imread(name)
	m=s.shape[0]
	n=s.shape[1]
	med=[[-9 for i in range(m)] for j in range(n)]
	for i in range(1, m-1):
		for j in range(1, n-1):
			l=[s[i-1][j-1][0], s[i-1][j][0], s[i-1][j+1][0], s[i][j-1][0], s[i][j][0], s[i][j+1][0], s[i+1][j-1][0], s[i+1][j][0], s[i+1][j+1][0]]
			l.sort()
			med[i][j]=l[4]

	for j in range(1, n-1):
		l1=[s[0][j-1][0], s[0][j+1][0], s[1][j][0], s[0][j][0], s[1][j-1][0], s[1][j+1][0]]
		l1.sort()
		med[0][j]=l1[2]

	for j in range(1, n-1):
		l1=[s[n-1][j-1][0], s[n-1][j+1][0], s[n-1][j][0], s[n-2][j][0], s[n-2][j-1][0], s[n-2][j+1][0]]
		l1.sort()
		med[n-1][j]=l1[2]

	for j in range(1, m-1):
		l1=[s[j-1][0][0], s[j+1][0][0], s[j][0][0], s[j][1][0], s[j-1][1][0], s[j+1][1][0]]
		l1.sort()
		med[j][0]=l1[2]

	for j in range(1, m-1):
		l1=[s[j-1][m-1][0], s[j+1][m-1][0], s[j][m-1][0], s[j][m-2][0], s[j-1][m-2][0], s[j+1][m-2][0]]
		l1.sort()
		med[j][m-1]=l1[2]

	l1=[s[0][0][0], s[0][1][0], s[1][0][0], s[1][1][0]]
	l1.sort()
	med[0][0]=int(l1[2])

	l1=[s[0][m-1][0], s[0][m-2][0], s[1][m-1][0], s[1][m-2][0]]
	l1.sort()
	med[0][m-1]=int(l1[2])

	l1=[s[n-1][0][0], s[n-1][1][0], s[n-2][0][0], s[n-2][1][0]]
	l1.sort()
	med[n-1][0]=int(l1[2])

	l1=[s[n-1][m-1][0], s[n-1][m-2][0], s[n-2][m-1][0], s[n-2][m-2][0]]
	l1.sort()
	med[n-1][m-1]=int(l1[2])
	return(med)


	
salt_and_pepper_noise(5, "sp_image_1.png", 0.1)
salt_and_pepper_noise(10, "sp_image_2.png", 0.2)
salt_and_pepper_noise(15, "sp_image_3.png", 0.3)
salt_and_pepper_noise(20, "sp_image_4.png", 0.4)
salt_and_pepper_noise(25, "sp_image_5.png", 0.5)
salt_and_pepper_noise(30, "sp_image_6.png", 0.6)
salt_and_pepper_noise(30, "sp_image_7.png", 0.7)
salt_and_pepper_noise(30, "sp_image_8.png", 0.8)
salt_and_pepper_noise(30, "sp_image_9.png", 0.9)
salt_and_pepper_noise(30, "sp_image_10.png", 1.0)


c=med_filter("sp_image_1.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out1_3.jpg", out)

c=med_filter("sp_image_1.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out1_5.jpg", out)

c=med_filter("sp_image_1.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out1_7.jpg", out)


c=med_filter("sp_image_2.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out2_3.jpg", out)

c=med_filter("sp_image_2.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out2_5.jpg", out)

c=med_filter("sp_image_2.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out2_7.jpg", out)


c=med_filter("sp_image_3.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out3_3.jpg", out)

c=med_filter("sp_image_3.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out3_5.jpg", out)

c=med_filter("sp_image_3.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out3_7.jpg", out)


c=med_filter("sp_image_4.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out4_3.jpg", out)

c=med_filter("sp_image_4.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out4_5.jpg", out)

c=med_filter("sp_image_4.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out4_7.jpg", out)


c=med_filter("sp_image_5.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out5_3.jpg", out)

c=med_filter("sp_image_5.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out5_5.jpg", out)

c=med_filter("sp_image_5.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out5_7.jpg", out)


c=med_filter("sp_image_6.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out6_3.jpg", out)

c=med_filter("sp_image_6.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out6_5.jpg", out)

c=med_filter("sp_image_6.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out6_7.jpg", out)


c=med_filter("sp_image_7.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out7_3.jpg", out)

c=med_filter("sp_image_7.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out7_5.jpg", out)

c=med_filter("sp_image_7.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out7_7.jpg", out)


c=med_filter("sp_image_8.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out8_3.jpg", out)

c=med_filter("sp_image_8.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out8_5.jpg", out)

c=med_filter("sp_image_8.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out8_7.jpg", out)


c=med_filter("sp_image_9.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out9_3.jpg", out)

c=med_filter("sp_image_9.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out9_5.jpg", out)

c=med_filter("sp_image_9.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out9_7.jpg", out)


c=med_filter("sp_image_10.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out10_3.jpg", out)

c=med_filter("sp_image_10.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out10_5.jpg", out)

c=med_filter("sp_image_10.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out10_7.jpg", out)