import numpy as np
from ipdb import set_trace

def get_gradient(f,x,eplison=0.0001,show=False):
	res =  (f(x+eplison)-f(x-eplison))/(2*eplison)
	if(show):
		from matplotlib.pyplot import plot,show
		plot(x,f(x))
		plot(x,res)
		show()

	return res
def compute_gradient(f,x,eplison=0.00001,show=False):
	gradient=np.zeros(len(x))
	for ii in range(len(x)):
		tmp=list(x)
		tmp2=list(x)
		tmp[ii] +=eplison
		tmp2[ii]  -=eplison
		gradient[ii]=(f(tmp)-f(tmp2))/(2*eplison)
		return gradient
def compute_hessian(f,x,eplison=0.00001):
	hessian=np.zeros([len(x),len(x)])
	for ii in np.arange(len(x)):
		for jj in np.arange(len(x)):
			xx=np.array(x,dtype='float64')
			xx[ii]+=eplison
			f_x=f(xx)
			#print xx,f_x
			xx[jj]+=eplison
			f_x_y=f(xx)
			#print xx,f_x_y
			xx=np.array(x,dtype='float64')
			xx[jj]+=eplison
			f_y=f(xx)
			#print f_y,xx
			rsl=(f_x_y+f(x)-f_x-f_y)/(eplison**2)
			#print rsl
			hessian[ii][jj]= rsl

	return hessian

def forward_spread(x,matrixs):
	xx=np.array(x)
	layers_output=[]
	layers_output.append(xx)
	for ii in matrixs:
		
		#set_trace()	        
		xx=np.dot(ii,xx)
		layers_output.append(xx)
	return layers_output
def cal_last_layer(y,a):
	y=np.array(y,dtype='float64')
	a=np.array(a,dtype='float64')
	return -(y-a)*(a*(1-a))

def cal_hidden_layer(a_s,delta,matrixs):
	result=[]
	delta_i=np.array(delta,dtype='float64')
	result.insert(0,delta_i)
	for ii in range(len(matrixs)-1,0,-1):
		#set_trace()
		#print delta_i , matrixs[ii]
		delta_i=np.dot((matrixs[ii].T),delta_i)

		delta_i=delta_i*(a_s[ii]*(1-a_s[ii]))
		result.insert(0,delta_i)

	return result
def update_wb(w,bb,alpha,delta_s,a_s):
	#w=np.array(ww,dtype='float64')
	for ii in range(len(w)):
		#set_trace()
		print w[ii],delta_s[ii],a_s[ii]
		w[ii]=w[ii]-alpha*np.dot(delta_s[ii].T,a_s[ii])
		bb[ii]=bb[ii]-alpha*delta_s[ii]


def check_all():
	matrixs=[]
	matrixs.append(np.random.random(4,2))
	matrixs.append(np.random.random(1,4))
	bb=[1,1]
	datas=[([1,2.0],1),([100,100],0)]
	a=forward_spread(datas[0][0],matrixs)
	delta=cal_last_layer(data[0][1],a)
	delta_s=cal_hidden_layer(a,delta,matrixs)
	update_wb(matrixs,bb,0.1,delta_s,a)









             