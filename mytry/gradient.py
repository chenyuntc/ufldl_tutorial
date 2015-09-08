#coding:utf8
import numpy as np
from numpy import array,random,dot,transpose
from scipy import optimize as opt
import scipy as sp
import pylab
from scipy.misc import imshow
#%pylab inline

def comput_gra(f,x,eplison=0.0001):
    fx=f(x)
    xx=x.reshape([-1,1]).repeat(x.shape[0],axis=1)
    ep=np.diag(np.repeat([eplison],x.shape[0]))
    #print f(xx-ep)
    s= (f(xx+ep)-f(xx-ep))/(2*eplison)
    grad=np.diag(s)
    return grad

def simple_function(x):
    f=3*x**2+2*x+4+5*x**3
    grad  = 6*x + 2 + 15 * x**2
    return f
def simple_function_grad(x):
     grad  = 6*x + 2 + 15 * x**2
     return grad
 if __name__ == '__main__':
 	g=comput_gra(simple_function,array([1,2,3,4]))-simple_function_grad(array([1,2,3,4]))
 	print g<0.0001