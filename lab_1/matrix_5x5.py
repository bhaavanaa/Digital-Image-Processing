### Read a matrix of size 5x5 and find the following using user defined functions- 1) sum, 2) max, 3) mean, 4) median, 5) mode, 6) stdDev, 7) freq. dist.


## Function for finding the sum
def Sum(p):					
	s=0
	for i in range(0, 5):
		for j in range(0, 5):
			s=s+p[i][j]
	return(s)


## Function for finding the max
def Max(p):					
	m=p[0][0]
	for i in range(0, 5):
		for j in range(0, 5):
			if(m<p[i][j]):
				m=p[i][j]
	return(m)


## Function for finding the mean
def Mean(p):
	return(p/25)


## Function for finding the median
def Median(p):
	arr=[]
	for i in range(0, 5):
		for j in range(0, 5):
			arr.append(p[i][j])
	arr.sort()
	return(arr[12])


## Function for finding the mode
def Mode(p):
	d={}
	for i in range(0, 5):
		for j in range(0, 5):
			if(p[i][j] in d):
				d[p[i][j]]=d[p[i][j]]+1
			else:
				d[p[i][j]]=1
	m1=max(d.values())
	arr=[key for key in d if d[key]==m1]
	return(d, arr)


## Function for finding the mode
def StdDev(p, m):
	s=0
	for i in range(0, 5):
		for j in range(0, 5):
			s=s+(p[i][j]-m)*(p[i][j]-m)
	s1=s/25
	return(s1**(1/2))


## Taking the 5x5 matrix as input
matrix=[]					
for i in range(0, 5):
	a=[]
	for j in range(0, 5):
		s="element ("+str(i+1)+", "+str(j+1)+"): "
		a.append(int(input(s)))
	matrix.append(a)
print("matrix: ", matrix)
print("\n")


## Sum
sum1=Sum(matrix)
print("sum is: ", sum1)


## Max
max1=Max(matrix)
print("max is: ", max1)


## Mean
mean1=Mean(sum1)
print("mean is: ", round(mean1, 3))


## Median
median1=Median(matrix)
print("median is: ", median1)


## Mode
d1, mode1=Mode(matrix)
print("mode is: ", mode1)


## Standard deviation
stdDev1=StdDev(matrix, mean1)
print("stdDev is: ", round(stdDev1, 3))


## Frequency distribution
print("frequency distribution is: ")
for i in d1:
	print(i, " - ", d1[i])
print("\n")
