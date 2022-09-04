import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
import mat73
import statistics
import math


# Open mat file & isolating the matrix field
matrix1 = scipy.io.loadmat('500.mat')
frames1 = matrix1['ans']
frames1 = frames1[1:,:,:] ## Delet the first row

matrix2 = scipy.io.loadmat('1000.mat')
frames2 = matrix2['ans']
frames2 = frames2[1:,:,:] ## Delet the first row

matrix3 = scipy.io.loadmat('1500.mat')
frames3 = matrix3["ans"]
frames3 = frames3[1:,:,:] ## Delet the first row

matrix4 = scipy.io.loadmat('750.mat')
frames4 = matrix4["ans"]
frames4 = frames4[1:,:,:] ## Delet the first row

f1_m = frames1[:,:,0]
f2_m = frames2[:,:,0]
f3_m = frames3[:,:,0]
f4_m = frames4[:,:,0]
  
# plt.plot(f4_m[500,:])

avg1 = f4_m.mean()
std_1 = (np.std(f4_m) * 100)/avg1
print('original mat STD:', std_1,'%')
################################################################################
############################# Dead pixel matrix ################################

f1 = f2_m
f2 = f1_m

f3 = f2 - f1

i = 0
k = 0

while i < 1023:
    while k < 1280:
        if f3[i,k] == 0:
            f3[i,k] = 0
            k = k + 1
        else:
            f3[i,k] = 1
            k = k + 1
    k = 0
    i = i + 1   

f_dead = f3
# res = np.argwhere(f3==0) 
# print(res)


#################################################################################
############################## Avreging 50 frames ###############################

def average_point(point,black_lvl):
   
    index = 49
    sum_frame = np.zeros((1023,1280), np.int16)

    while index >= 0:
          
          sum_frame = sum_frame + point[:,:,index] - black_lvl
          index = index - 1

    I = (sum_frame)/50

    return I
    
I1 = average_point(frames1,300)    
I2 = average_point(frames2,300)    
I3 = average_point(frames3,300)    
I4 = average_point(frames4,300)    
    
#################################################################################
############################## Dead pixel corection #############################

def avg_dead_pix(mat_x):
    i = 0
    k = 0
    mat = mat_x * f_dead
    avg = mat.mean()
    while i < 1023:
        while k < 1280:
            if mat[i,k] == 0:
                mat[i,k] = avg
            k = k + 1
        k = 0
        i = i + 1
    return mat

I1_no_dead = avg_dead_pix(I1)
I2_no_dead = avg_dead_pix(I2)
I3_no_dead = avg_dead_pix(I3)
I4_no_dead = avg_dead_pix(I4)

# res = np.argwhere(mat4_no_dead==0) 
# print(res)       
    
#################################################################################
################################ Corect BAD pix   ###############################

ones = np.ones((1023,1280)).astype(int)

def bad_pix(mat):
    a1 = 0 + 93
    a2 = a1 + 93
    b1 = 0
    b2 = b1 + 128
    i = a1
    k = b1


    while a1 < 1023:
        while b1 < 1280: 
              avg = mat[a1:a2,b1:b2].mean()
              while i < a2:
                  while k < b2:
                      if mat[i,k] > 1.02*avg or mat[i,k] < 0.98*avg:
                          mat[i,k] = avg
                          ones[i,k] = 0
                      k = k + 1
                  k = b1
                  i = i + 1
              k = b1
              i = a1
              # avg_l.append(avg)
              b1 = b1 + 128
              b2 = b2 + 128
        i = a1
        k = b1
        b1 = 0
        b2 = b1 + 128     
        a1 = a1 + 93
        a2 = a2 + 93
    return mat
    
I1_final = bad_pix(I1_no_dead)
I2_final = bad_pix(I2_no_dead)
I3_final = bad_pix(I3_no_dead)    
I4_final = bad_pix(I4_no_dead)    

# plt.plot(I2_final[500,:])

avg2 = I4_final.mean()
std_2 = (np.std(I4_final) * 100)/avg2
print('no bad pix mat STD:', std_2,'%')
#################################################################################
################################ Flat 2D to 1D  #################################

I1_1d = I1_final.flatten().astype(int)
I2_1d = I2_final.flatten().astype(int)
I3_1d = I3_final.flatten().astype(int)    
I4_1d = I4_final.flatten().astype(int)     
    
#################################################################################
############################# Liniar regretion  #################################

####Execute a method that returns some important key values of Linear Regression
def Y_index(list_of_num, x_num):
     
      slope, intercept, r, p, std_err = stats.linregress(axis, list_of_num)
      
#### Func that return Y based on leniar regretion graph
      def myfunc(x):
        return slope * x + intercept

      # mymodel = list(map(myfunc, axis))
      y = myfunc(x_num)
      return y


##################################################################################
#################################### Main ########################################

# fun that crate temperari list of pix data from I1,I2,I3
def temp_list(i):
      temp = list()
      temp = (I1_1d[i], I2_1d[i], I3_1d[i])
      return temp 

axis = [500,1000,1500] ## 3 poin power
corection1 = list()
corection2 = list()
ind = 0

while ind < 1309440:
    temp = temp_list(ind)
    y1 = Y_index(temp,500)
    y2 = Y_index(temp,1500)   
    m = (y2-y1)/(1000)    
    c = (500 * y2 - y1 * 1500)/(1000)
    corection1.append(m)
    corection2.append(c)
    ind = ind + 1
 
    
     

mm = np.array(corection1)  
cc = np.array(corection2) 
mm_mat = mm.reshape((1023,1280))
cc_mat = cc.reshape((1023,1280))

I4m = I4_1d.reshape((1023,1280))

avg_mm = mm_mat.mean()
avg_cc = cc_mat.mean()

a = avg_mm/mm_mat
b = avg_cc/cc_mat

corect = a * I4_final + b
corect = corect.astype(int)
avg3 = corect.mean()
std_3 = np.std(corect)

# print('corected mat STD:', std_3)
# print('corected mat avg:', avg3)
plt.plot(corect[500,:])    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    