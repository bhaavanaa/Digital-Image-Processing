import numpy as np
import cmath
from math import pi
import numpy as np
import cv2
from PIL import Image
from sympy import ifft 


def DFT(x):
    M = x.shape[0]
    u = np.arange(M)
    k = u.reshape((M, 1))
    exponentTerm = np.exp(-2j * np.pi * k * u / M)
    return np.dot(x, exponentTerm)


def FFT_1D(x):
    M = x.shape[0]
    if M==1 :
        return DFT(x)
    else:
        fftOfEvenSamples = FFT_1D(x[::2])
        fftOfOddSamples = FFT_1D(x[1::2])
        exponentTerm = np.exp(-2j * np.pi * np.arange(M) / M)
        return list(np.concatenate([fftOfEvenSamples + exponentTerm[:int(M/2)] *fftOfOddSamples, fftOfEvenSamples + exponentTerm[int(M/2):] * fftOfOddSamples]))


def FFT_2D(mat, rows, cols):
	mat1=[[0 for i in range(cols)] for j in range(rows)]
	mat2=[[0 for i in range(rows)] for j in range(cols)]
	for i in range(rows):
		mat1[i]=FFT_1D(np.array(mat[i]))
	matr=np.transpose(mat1)
	for i in range(cols):
		mat2[i]=FFT_1D(np.array(matr[i]))
	return(np.transpose(mat2))


def IDFT_1D(arr):
	if(len(arr)<=1):
		return(arr)
	else:
		arre=[]
		arro=[]
		for i in range(len(arr)):
			if(i%2==0):
				arre.append(arr[i])
			else:
				arro.append(arr[i])
		G=IDFT_1D(arre)
		H=IDFT_1D(arro)
		FL=[0 for i in range(int(len(arr)/2))]
		FR=[0 for i in range(int(len(arr)/2))]
		T=cmath.exp(2*cmath.pi*1j/len(arr))
		for i in range(int(len(arr)/2)):
			FL[i]=(G[i]+T**i*H[i])
			FR[i]=(G[i]-T**i*H[i])
		return(FL+FR)


def IDFT_2D(mat, rows, cols):
	mat1=[[0 for i in range(cols)] for j in range(rows)]
	mat2=[[0 for i in range(rows)] for j in range(cols)]
	mat3=[[0 for i in range(rows)] for j in range(cols)]
	for i in range(rows):
		mat1[i]=IDFT_1D(mat[i])
	matr=np.transpose(mat1)
	for i in range(cols):
		mat2[i]=IDFT_1D(matr[i])
	for i in range(rows):
		for j in range(cols):
			mat3[i][j]=mat2[i][j]/(rows*cols)
	return(np.transpose(mat3))



imgl = cv2.imread("lena.png", 0)			#reading lena
ml=imgl.shape[0]
nl=imgl.shape[1]


imgd1 = cv2.imread("dog.png")				#reading dog
new_size=(512, 512)
imgd2=cv2.resize(imgd1, new_size)
cv2.imwrite("new_dog.png", imgd2)
imgd=cv2.imread("new_dog.png", 0)
md=imgd.shape[0]
nd=imgd.shape[1]


lena1=FFT_2D(imgl, ml, nl)

phase_l=[[0 for i in range(ml)] for j in range(nl)]		#finding the phase and magnitude of lena
mag_l=[[0 for i in range(ml)] for j in range(nl)]
for i in range(ml):
	for j in range(nl):
		mag_l[i][j]=np.absolute(lena1[i][j])
		phase_l[i][j]=np.angle(lena1[i][j])


dog1=FFT_2D(imgd, md, nd)								#finding the phase and magnitude of dog

phase_d=[[0 for i in range(md)] for j in range(nd)]
mag_d=[[0 for i in range(md)] for j in range(nd)]

for i in range(md):
	for j in range(nd):
		mag_d[i][j]=np.absolute(dog1[i][j])
		phase_d[i][j]=np.angle(dog1[i][j])


magd_phl=[[0 for i in range(md)] for j in range(nd)]	#combining magnitude of lena and phase of dog
for i in range(md):
	for j in range(nd):
		magd_phl[i][j]=mag_d[i][j]* cmath.exp(1j*phase_l[i][j])

magd_phl_im1= IDFT_2D(magd_phl, ml, nl) 
imgUMat = np.float32(magd_phl_im1)
cv2.imwrite("magd_phl1.png", imgUMat)


magl_phd=[[0 for i in range(md)] for j in range(nd)]	#combining magnitude of dog and phase of lena
for i in range(md):
	for j in range(nd):
		magl_phd[i][j]=mag_l[i][j]* cmath.exp(1j*phase_d[i][j])

magl_phd_im1= IDFT_2D(magl_phd, ml, nl) 
imgUMat1 = np.float32(magl_phd_im1)
cv2.imwrite("magl_phd1.png", imgUMat1)
