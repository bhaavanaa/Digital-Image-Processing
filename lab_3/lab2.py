import cv2
import numpy as np
from skimage.util import random_noise


img = cv2.imread("lena.png")
#cv2.imshow('original image',img)


X=img.shape[0]	#x coordinate
Y=img.shape[1]	#y coordinate


def bilinear_ipoo(p, q, new_lena):
	m1=[[1, p-1, q-1, (p-1)*(q-1)], [1, p+1, q+1, (p+1)*(q+1)], [1, p+1, q-1, (p+1)*(q-1)], [1, p-1, q+1, (p-1)*(q+1)]]
	A=np.array(m1)
	m20=[new_lena[p-1][q-1][0], new_lena[p+1][q+1][0], new_lena[p+1][q-1][0], new_lena[p-1][q+1][0]]
	m21=[new_lena[p-1][q-1][1], new_lena[p+1][q+1][1], new_lena[p+1][q-1][1], new_lena[p-1][q+1][1]]
	m22=[new_lena[p-1][q-1][2], new_lena[p+1][q+1][2], new_lena[p+1][q-1][2], new_lena[p-1][q+1][2]]
	B0=np.array(m20)
	B1=np.array(m21)
	B2=np.array(m22)
	if(np.linalg.det(A)!=0):
		print("line24",p,q,end=' ')
		X0=np.linalg.inv(A).dot(B0)
		X1=np.linalg.inv(A).dot(B1)
		X2=np.linalg.inv(A).dot(B2)
		c0=X0[0]+X0[1]*q+X0[2]*p+X0[3]*p*q
		c1=X1[0]+X1[1]*q+X1[2]*p+X1[3]*p*q
		c2=X2[0]+X2[1]*q+X2[2]*p+X2[3]*p*q
	else:
		c0= new_lena[p-1][q-1][0]/4+ new_lena[p+1][q+1][0]/4+ new_lena[p+1][q-1][0]/4+new_lena[p-1][q+1][0]/4
		c1= new_lena[p-1][q-1][1]/4+ new_lena[p+1][q+1][1]/4+ new_lena[p+1][q-1][1]/4+ new_lena[p-1][q+1][1]/4
		c2= new_lena[p-1][q-1][2]/4+ new_lena[p+1][q+1][2]/4+ new_lena[p+1][q-1][2]/4+ new_lena[p-1][q+1][2]/4
	if(c0<0):
		print("line 35",p,q)
	return(c0, c1, c2)
	
def bilinear_ipeo(p, q, new_lena):
	#print("32 line",p,q)
	m1=[[1, p, q-1, (p)*(q-1)], [1, p, q+1, (p)*(q+1)], [1, p-1, q, (p-1)*(q)], [1, p+1, q, (p+1)*(q)]]
	A=np.array(m1)
	m20=[new_lena[p][q-1][0], new_lena[p][q+1][0], new_lena[p-1][q][0], new_lena[p+1][q][0]]
	m21=[new_lena[p][q-1][1], new_lena[p][q+1][1], new_lena[p-1][q][1], new_lena[p+1][q][1]]
	m22=[new_lena[p][q-1][2], new_lena[p][q+1][2], new_lena[p-1][q][2], new_lena[p+1][q][2]]
	B0=np.array(m20)
	B1=np.array(m21)
	B2=np.array(m22)
	if(np.linalg.det(A)!=0):
		X0=np.linalg.inv(A).dot(B0)
		X1=np.linalg.inv(A).dot(B1)
		X2=np.linalg.inv(A).dot(B2)
		c0=X0[0]+X0[1]*q+X0[2]*p+X0[3]*p*q
		c1=X1[0]+X1[1]*q+X1[2]*p+X1[3]*p*q
		c2=X2[0]+X2[1]*q+X2[2]*p+X2[3]*p*q
	else:
		c0= new_lena[p][q-1][0]/4+ new_lena[p][q+1][0]/4+ new_lena[p-1][q][0]/4+new_lena[p+1][q][0]/4
		c1= new_lena[p][q-1][1]/4+ new_lena[p][q+1][1]/4+ new_lena[p-1][q][1]/4+ new_lena[p+1][q][1]/4
		c2= new_lena[p][q-1][2]/4+ new_lena[p][q+1][2]/4+ new_lena[p-1][q][2]/4+ new_lena[p+1][q][2]/4
	
	return(c0, c1, c2)
	
def bilinear_ipcorfc(p, q, new_lena):
	m1=[[1, p-1, q, (p-1)*(q)], [1, p+1, q, (p+1)*(q)], [1, p, q+1, (p)*(q+1)]]
	A=np.array(m1)
	c0= new_lena[p-1][q][0]/3+ new_lena[p+1][q][0]/3+ new_lena[p][q+1][0]/3
	c1= new_lena[p-1][q][1]/3+ new_lena[p+1][q][1]/3+ new_lena[p][q+1][1]/3
	c2=new_lena[p-1][q][2]/3+ new_lena[p+1][q][2]/3+ new_lena[p][q+1][2]/3
	
	return(c0, c1, c2)

def bilinear_ipcorlc(p, q, new_lena):
	m1=[[1, p-1, q, (p-1)*(q)], [1, p+1, q, (p+1)*(q)], [1, p, q-1, (p)*(q-1)]]
	A=np.array(m1)
#	m20=[new_lena[p-1][q][0], new_lena[p+1][q][0], new_lena[p][q-1][0]]
#	m21=[new_lena[p-1][q][1], new_lena[p+1][q][1], new_lena[p][q-1][1]]
#	m22=[new_lena[p-1][q][2], new_lena[p+1][q][2], new_lena[p][q-1][2]]
#	B0=np.array(m20)
#	B1=np.array(m21)
#	B2=np.array(m22)
#	X0=np.linalg.solve(A, B0)
#	X1=np.linalg.solve(A, B1)
#	X2=np.linalg.solve(A, B2)
#	c0=X0[0]+X0[1]*q+X0[2]*p+X0[3]*p*q
#	c1=X1[0]+X1[1]*q+X1[2]*p+X1[3]*p*q
#	c2=X2[0]+X2[1]*q+X2[2]*p+X2[3]*p*q
	
	c0=new_lena[p-1][q][0]/3+ new_lena[p+1][q][0]/3+ new_lena[p][q-1][0]/3
	c1=new_lena[p-1][q][1]/3+ new_lena[p+1][q][1]/3+ new_lena[p][q-1][1]/3
	c2=new_lena[p-1][q][2]/3+ new_lena[p+1][q][2]/3+ new_lena[p][q-1][2]/3
	return(c0, c1, c2)
	
def bilinear_ipcorfr(p, q, new_lena):
	m1=[[1, p+1, q, (p+1)*(q)], [1, p, q+1, (p)*(q+1)], [1, p, q-1, (p)*(q-1)]]
	A=np.array(m1)
#	m20=[new_lena[p+1][q][0], new_lena[p][q+1][0], new_lena[p][q-1][0]]
#	m21=[new_lena[p+1][q][1], new_lena[p][q+1][1], new_lena[p][q-1][1]]
#	m22=[new_lena[p+1][q][2], new_lena[p][q+1][2], new_lena[p][q-1][2]]
#	B0=np.array(m20)
#	B1=np.array(m21)
#	B2=np.array(m22)
#	X0=np.linalg.solve(A, B0)
#	X1=np.linalg.solve(A, B1)
#	X2=np.linalg.solve(A, B2)
#	c0=X0[0]+X0[1]*q+X0[2]*p+X0[3]*p*q
#	c1=X1[0]+X1[1]*q+X1[2]*p+X1[3]*p*q
#	c2=X2[0]+X2[1]*q+X2[2]*p+X2[3]*p*q

	c0=new_lena[p+1][q][0]/3+ new_lena[p][q+1][0]/3+ new_lena[p][q-1][0]/3
	c1=new_lena[p+1][q][1]/3+ new_lena[p][q+1][1]/3+ new_lena[p][q-1][1]/3
	c2=new_lena[p+1][q][2]/3+ new_lena[p][q+1][2]/3+ new_lena[p][q-1][2]/3
	
	return(c0, c1, c2)
	
def bilinear_ipcorlr(p, q, new_lena):
	m1=[[1, p-1, q, (p-1)*(q)], [1, p, q+1, (p)*(q+1)], [1, p, q-1, (p)*(q-1)]]
	A=np.array(m1)
#	m20=[new_lena[p-1][q][0], new_lena[p][q+1][0], new_lena[p][q-1][0]]
#	m21=[new_lena[p-1][q][1], new_lena[p][q+1][1], new_lena[p][q-1][1]]
#	m22=[new_lena[p-1][q][2], new_lena[p][q+1][2], new_lena[p][q-1][2]]
#	B0=np.array(m20)
#	B1=np.array(m21)
#	B2=np.array(m22)
#	X0=np.linalg.solve(A, B0)
#	X1=np.linalg.solve(A, B1)
#	X2=np.linalg.solve(A, B2)
#	c0=X0[0]+X0[1]*q+X0[2]*p+X0[3]*p*q
#	c1=X1[0]+X1[1]*q+X1[2]*p+X1[3]*p*q
#	c2=X2[0]+X2[1]*q+X2[2]*p+X2[3]*p*q
	c0=new_lena[p-1][q][0]/3+ new_lena[p][q+1][0]/3+ new_lena[p][q-1][0]/3
	c1=new_lena[p-1][q][1]/3+new_lena[p][q+1][1]/3+ new_lena[p][q-1][1]/3
	c2=new_lena[p-1][q][2]/3+new_lena[p][q+1][2]/3+ new_lena[p][q-1][2]/3
	return(c0, c1, c2)


def new_image(sf):
	new_x=X*sf
	new_y=Y*sf
	rows, cols, d=(new_x, new_y, 3)
	new_lena=[[[-1 for i in range(d)] for j in range(cols)] for k in range(rows)]
	#print(new_lena)
	
	for i in range(X):					#even-even
		for j in range(Y):
			new_lena[sf*i][sf*j][0]=img[i][j][0]
			new_lena[sf*i][sf*j][1]=img[i][j][1]
			new_lena[sf*i][sf*j][2]=img[i][j][2]
	#print(new_lena)
	output0=new_lena
	output_0 = np.array(output0, dtype = 'uint8')
	cv2.imwrite("o0.jpg",output_0)

	flat_lena=[]
	for sub1 in new_lena:
		for sub2 in sub1:
			flat_lena.append(sub2)
	if([-1, -1, -1] in flat_lena):	
		for i in range(1, new_x-1):			#odd-odd
			for j in range(1, new_y-1):
				if(i%2==1 and j%2==1):
					new_lena[i][j][0], new_lena[i][j][1], new_lena[i][j][2]=bilinear_ipoo(i, j, new_lena)

		output1=new_lena
		output_1 = np.array(output1, dtype = 'uint8')
		cv2.imwrite("o1.jpg",output_1)

		for i in range(1, new_x-1):			#even-odd
			for j in range(1, new_y-1):
				if(i%2==0 and j%2==1 and new_lena[i][j]==[-1, -1, -1]):
					new_lena[i][j][0], new_lena[i][j][1], new_lena[i][j][2]=bilinear_ipeo(i, j, new_lena)
		for i in range(1, new_x-1):			#odd-even
			for j in range(1, new_y-1):
				if(i%2==1 and j%2==0 and new_lena[i][j]==[-1, -1, -1]):
					new_lena[i][j][0], new_lena[i][j][1], new_lena[i][j][2]=bilinear_ipeo(i, j, new_lena)
		for i in range(1, new_x-1):
			if(new_lena[i][0]==[-1,-1,-1]):
				new_lena[i][0][0], new_lena[i][0][1], new_lena[i][0][2]=bilinear_ipcorfc(i, 0, new_lena)
		for i in range(1, new_x-1):
			if(new_lena[i][new_y-1]==[-1,-1,-1]):
				new_lena[i][new_y-1][0], new_lena[i][new_y-1][1], new_lena[i][new_y-1][2]=bilinear_ipcorlc(i, new_y-1, new_lena)
		for i in range(1, new_y-1):
			if(new_lena[i][new_y-1]==[-1,-1,-1]):
				new_lena[0][i][0], new_lena[0][i][1], new_lena[0][i][2]=bilinear_ipcorfr(0, i, new_lena)
		for i in range(1, new_y-1):
			if(new_lena[i][new_y-1]==[-1,-1,-1]):
				new_lena[new_x-1][i][0], new_lena[new_x-1][i][1], new_lena[new_x-1][i][2]=bilinear_ipcorlr(new_x-1, i, new_lena)
		print(new_lena)
			
	return(new_lena)
	
#scale_2_output=new_image(1)
scale_2_output1=new_image(2)
scale_2_output = np.array(scale_2_output1, dtype = 'uint8')
#scale_2_output=new_image(0.5)
print(scale_2_output)
cv2.imwrite("scale_2_lena.jpg", scale_2_output)
	
res = cv2.resize(img,None,fx=2, fy=2, interpolation =  cv2.INTER_LINEAR)

cv2.imwrite("scale_2_lena_actual.jpg", res)

