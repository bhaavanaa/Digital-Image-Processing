import cv2
import numpy as np

img=cv2.imread("lenabw.jpeg")
img_eye=cv2.imread("lena_eye.jpeg")
# img1=cv2.imread("1.png")
# img2=cv2.imread("2.png")
# img3=cv2.imread("3.png")
# img4=cv2.imread("4.png")
# img5=cv2.imread("5.png")
# img6=cv2.imread("6.png")
# img7=cv2.imread("7.png")
# img9=cv2.imread("9.png")

#print(img1, img2)
m=img.shape[0]
n=img.shape[1]

a=img_eye.shape[0]
b=img_eye.shape[1]

print(img[123][30],'---',img_eye[0][0])
#print(img.shape,img_eye.shape)

cor_mat=[[-9 for i in range(n+b-1)] for j in range(m+a-1)]
max1=0
max_posx=0
max_posy=0
for x in range(m+a-1):
	for y in range(n+b-1):
		sum2=0
		for j in range(int(-(b-1)/2), int((b-1)/2)):
			for i in range(int(-(a-1)/2), int((a-1)/2)):
				#print("here", x, y, j, i)
				if(x+i<225 and y+j<225):
					sum2=sum2+img[x+i][y+j][0]*img_eye[i][j][0]
				
		if(max1<=sum2):
			max1=sum2
			max_posx=x
			max_posx=y
		print(x, y, j, i, m, a, n, b,sum2, max_posx,max_posy)
		cor_mat[x][y]=sum2	
#out = np.array(cor_mat, dtype = 'uint8')
print(cor_mat[0][0], img_eye[0][0])
#cv2.imwrite("out.jpg", out)

max2=max(cor_mat)
print(max1)
print( max_posx, max_posy)#img[max_posx][max_posy],img_eye[0][0]) 
print(np.amax(cor_mat))