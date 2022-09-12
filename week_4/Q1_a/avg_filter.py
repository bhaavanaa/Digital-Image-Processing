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
		noise_img = random_noise(i1, mode='s&p', amount=k)
		s=s+noise_img
	final_img = np.array(255*s/times, dtype = 'uint8')
	cv2.imwrite(f, final_img)


def avg_filter(n1, name):
	avg_fil=[[1/(n1*n1) for i in range(n1)] for j in range(n1)]
	#print(m)
	s=cv2.imread(name)
	m=s.shape[0]
	n=s.shape[1]
	print(s[0][0])
	#for x in range(0, )
	#print(m, n)
	a=n1
	b=n1
	cor_mat=[[-9 for i in range(m+a-1)] for j in range(n+b-1)]
	for x in range(m+a-1):
		for y in range(n+b-1):
			sum1=0
			for j in range(int(-(b-1)/2), int((b-1)/2)):
				sum2=0
				for i in range(int(-(a-1)/2), int((a-1)/2)):
					#print(x, y, j, i)
					if(x+i<256 and y+j<256):
						sum2=sum2+s[x+i][y+j][0]*avg_fil[i][j]
				sum1=sum1+sum2
			cor_mat[x][y]=int(sum1)
	#out = np.array(cor_mat, dtype = 'uint8')
	print(cor_mat[0][0], s[0][0])
	#cv2.imwrite("out.jpg", out)
	return(cor_mat)

	
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


c=avg_filter(3, "sp_image_1.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out1_3.jpg", out)

c=avg_filter(5, "sp_image_1.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out1_5.jpg", out)

c=avg_filter(7, "sp_image_1.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out1_7.jpg", out)


c=avg_filter(3, "sp_image_2.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out2_3.jpg", out)

c=avg_filter(5, "sp_image_2.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out2_5.jpg", out)

c=avg_filter(7, "sp_image_2.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out2_7.jpg", out)


c=avg_filter(3, "sp_image_3.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out3_3.jpg", out)

c=avg_filter(5, "sp_image_3.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out3_5.jpg", out)

c=avg_filter(7, "sp_image_3.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out3_7.jpg", out)


c=avg_filter(3, "sp_image_4.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out4_3.jpg", out)

c=avg_filter(5, "sp_image_4.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out4_5.jpg", out)

c=avg_filter(7, "sp_image_4.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out4_7.jpg", out)


c=avg_filter(3, "sp_image_5.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out5_3.jpg", out)

c=avg_filter(5, "sp_image_5.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out5_5.jpg", out)

c=avg_filter(7, "sp_image_5.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out5_7.jpg", out)


c=avg_filter(3, "sp_image_6.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out6_3.jpg", out)

c=avg_filter(5, "sp_image_6.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out6_5.jpg", out)

c=avg_filter(7, "sp_image_6.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out6_7.jpg", out)


c=avg_filter(3, "sp_image_7.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out7_3.jpg", out)

c=avg_filter(5, "sp_image_7.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out7_5.jpg", out)

c=avg_filter(7, "sp_image_7.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out7_7.jpg", out)


c=avg_filter(3, "sp_image_8.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out8_3.jpg", out)

c=avg_filter(5, "sp_image_8.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out8_5.jpg", out)

c=avg_filter(7, "sp_image_8.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out8_7.jpg", out)


c=avg_filter(3, "sp_image_9.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out9_3.jpg", out)

c=avg_filter(5, "sp_image_9.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out9_5.jpg", out)

c=avg_filter(7, "sp_image_9.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out9_7.jpg", out)


c=avg_filter(3, "sp_image_10.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out10_3.jpg", out)

c=avg_filter(5, "sp_image_10.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out10_5.jpg", out)

c=avg_filter(7, "sp_image_10.png")
out = np.array(c, dtype = 'uint8')
cv2.imwrite("out10_7.jpg", out)