import cv2


img1=cv2.imread("hdraw.png")
img0=cv2.imread("0.png")
img1=cv2.imread("1.png")
img2=cv2.imread("2.png")
img3=cv2.imread("3.png")
img4=cv2.imread("4.png")
img5=cv2.imread("5.png")
img6=cv2.imread("6.png")
img7=cv2.imread("7.png")
img9=cv2.imread("9.png")

#print(img1, img2)
m=img1.shape[0]
n=img1.shape[1]

a=img2.shape[0]
b=img2.shape[1]


cor_mat=[[-9 for i in range(n+b-1)] for j in range(m+a-1)]
max1=0
max_posx=0
max_posy=0
for x in range(m+a-1):
	for y in range(n+b-1):
		sum1=0
		for j in range(int(-(b-1)/2), int((b-1)/2)):
			sum2=0
			for i in range(int(-(a-1)/2), int((a-1)/2)):
				#print(x, y, j, i)
				if(x+i<256 and y+j<256):
					sum2=sum2+img1[x+i][y+j][0]*img2[i][j]
			sum1=sum1+sum2
			# if(max1<sum1[0]):
			# 	max1=sum1
			# 	max_posx=x
			# 	max_posx=y
		print(x, y, j, i, m, a, n, b)
		cor_mat[x][y]=sum1	
#out = np.array(cor_mat, dtype = 'uint8')
print(cor_mat[0][0], img1[0][0])
#cv2.imwrite("out.jpg", out)

max1=max(cor_mat)
print(max1)
print( x, y)
