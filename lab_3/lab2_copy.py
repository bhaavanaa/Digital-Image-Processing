import cv2
import numpy as np
from skimage.util import random_noise


img1 = cv2.imread("lena.png")
#cv2.imshow('original image',img)
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


X=img.shape[0]	#x coordinate
Y=img.shape[1]	#y coordinate


def bilinear_ip(p, q, img, sf):
	g1=p/sf-int(p/sf)
	g2=q/sf-int(q/sf)
	h1=int(p/sf)
	h2=int(q/sf)
	a1=g1*g2
	a2=g1*(1-g2)
	a3=(1-g1)*g2
	a4=(1-g1)*(1-g2)
	at=a4
	if int(p/sf)+1 < img.shape[0] and int(q/sf)+1 < img.shape[1]:
		at = at+a1+a2+a3
	elif int(p/sf)+1< img.shape[0]:
		at = at+a3
	elif int(q/sf)+1< img.shape[1]:
		at = at+a2

	at=a1+a2+a3+a4
	c0=((a1*img[h1+1][h2+1])/at)+((a2*img[h1+1][h2])/at)+((a3*img[h1][h2+1])/at)+((a4*img[h1][h2])/at)


	c0=a4*img[h1][h2]/at
	if int(p/sf)+1 < img.shape[0] and int(q/sf)+1 < img.shape[1]:
		c0+= (a1*img[h1+1][h2+1]/at)+(a2*img[h1+1][h2]/at)+(a3*img[h1][h2+1]/at)		
	elif int(p/sf)+1 < img.shape[0]:
		c0+= (a3*i mg[h1][h2+1]/at)
	elif int(q/sf)+1 < img.shape[1]:
		c0+= (a2*img[h1+1][h2]/at)

	#c1=((a1*new_img[h1+1][h2+1][1])/at)+((a2*new_lena[h1+1][h2][1])/at)+((a3*new_lena[h1][h2+1][1])/at)+((a4*new_lena[h1][h2][1])/at)
	#c2=((a1*new_lena[h1+1][h2+1][2])/at)+((a2*new_lena[h1+1][h2][2])/at)+((a3*new_lena[h1][h2+1][2])/at)+((a4*new_lena[h1][h2][2])/at)
	return(c0)
	
def new_image(sf):
	new_x=int(X*sf)
	new_y=int(Y*sf)
	rows, cols, d=(new_x, new_y, 1)
	#new_lena=[[[-1 for i in range(d)] for j in range(cols)] for k in range(rows)]
	new_lena=[[-1 for i in range(rows)] for j in range(cols)]
	#print(new_lena)
	
	for i in range(X):					#even-even
		for j in range(Y):
			new_lena[int(sf*i)][int(sf*j)]=img[i][j]
			#new_lena[sf*i][sf*j][1]=img[i][j][1]
			#new_lena[sf*i][sf*j][2]=img[i][j][2]

		
	for i in range(new_x-2):			#odd-odd
		for j in range(new_y-2):
			if(new_lena[i][j]==-1):
				new_lena[i][j]=bilinear_ip(i, j, img, sf)
	return(new_lena)
	
scale_1_output1=new_image(1)
scale_1_output = np.array(scale_1_output1, dtype = 'uint8')
cv2.imwrite("scale_1_lena1.jpg", scale_1_output)

scale_2_output1=new_image(2)
scale_2_output = np.array(scale_2_output1, dtype = 'uint8')
cv2.imwrite("scale_2_lena1.jpg", scale_2_output)

scale_half_output1=new_image(0.5)
scale_half_output = np.array(scale_half_output1, dtype = 'uint8')
print(scale_half_output)
cv2.imwrite("scale_half_lena1.jpg", scale_half_output)
	
res = cv2.resize(img,None,fx=2, fy=2, interpolation =  cv2.INTER_LINEAR)

cv2.imwrite("scale_2_lena_actual1.jpg", res)

