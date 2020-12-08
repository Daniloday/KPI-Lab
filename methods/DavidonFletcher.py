import os
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath('./functions'))

import Ackley
import Multimodal
import Quadratic

arrAckley = {
    "f": Ackley.f,
    "dfdx": Ackley.dfdx,
    "dfdy": Ackley.dfdy,
    "dfdxdx": Ackley.dfdxdx,
    "dfdydy": Ackley.dfdydy
}

arrMultimodal = {
    "f": Multimodal.f,
    "dfdx": Multimodal.dfdx,
    "dfdy": Multimodal.dfdy,
    "dfdxdx": Multimodal.dfdxdx,
    "dfdydy": Multimodal.dfdydy
}

arrQuadratic = {
    "f": Quadratic.f,
    "dfdx": Quadratic.dfdx,
    "dfdy": Quadratic.dfdy,
    "dfdxdx": Quadratic.dfdxdx,
    "dfdydy": Quadratic.dfdydy
}

arrFunc = {
    "Ackley": arrAckley,
    "Quadratic": arrQuadratic,
    "Multimodal": arrMultimodal
}

# def optimization(func_0, x_0, y_0, h_x0, h_y0, eps):
def f1 (x1, x2):
    return (-cos(x1)*cos(x2)*exp(-((x1-pi)**2+(x2-pi)**2))).n()
def f2 (x1, x2):
    return (x1 + 2*(x2**2) + exp(x1**2 + x2**2)).n()
def f3 (x1, x2):
    return 20 + x1**2 + x2**2 - 10*cos(2*pi*x1) - 10*cos(2*pi*x2)

def df (x, y, f):
    x1, x2 = symbols ('x1 x2')
    df1 = (diff (f(x1,x2), x1)).subs({x1:x,x2:y}).n()
    df2 = (diff (f(x1,x2), x2)).subs({x1:x,x2:y}).n()
    return np.array([df1, df2], dtype=np.float64)

def golden (x, p, f):
    l = symbols ('l')
    fl = f(x[0]-l*p[0],x[1]-l*p[1])
    a = 0.381966
    e0 = 0.001
    a0 = 0
    b0 = 1
    k1 = a0 + a*(b0-a0)
    k2 = a0 + b0 -k1  #b0 - a(b0-a0)
    f1 = fl.subs({l:k1})
    f2 = fl.subs({l:k2})
    while (b0-a0>e0):
        if (f1<=f2):
            b0=k2
            k2=k1
            f2=f1
            k1 = a0 + a*(b0-a0)
            f1 = fl.subs({l:k1})
        else:
            a0 = k1
            k1 = k2
            f1 = f2
            k2 = a0 + b0 -k1
            f2 = fl.subs({l:k2})
    lmd = (a0 + b0)/2
    return lmd
           
def Hessian (x, f):
    x1, x2 = symbols ('x1 x2')
    df1 = (diff (f(x1,x2), x1, x1)).subs({x1:x[0],x2:x[1]}).n()
    df2 = (diff (f(x1,x2), x1, x2)).subs({x1:x[0],x2:x[1]}).n()
    df3 = (diff (f(x1,x2), x2, x1)).subs({x1:x[0],x2:x[1]}).n()
    df4 = (diff (f(x1,x2), x2, x2)).subs({x1:x[0],x2:x[1]}).n()
    return np.array([[df1, df2],[df3, df4]], dtype=np.float64)

def dfp (x, f):
    fr = pd.DataFrame(columns=['X1', 'X2', 'F(x)', 'A', 'H_inv', 'lambda*p'])
    I=np.array([[1,0],
               [0,1]])
    A=I
    fr=fr.append({'X1':x[0], 'X2': x[1], 'F(x)': f(x[0], x[1]), 
                      'A': np.around(A,2), 
                      'lambda*p': '',
                      'H_inv': np.around(np.linalg.inv(Hessian(x, f)), 2)},
				ignore_index=True)
    while np.linalg.norm(df(x[0], x[1], f))>e:
        p = np.dot(A, df(x[0], x[1], f))
        x0 = x
        # минимизируем функцию f (xk - l*p)
        lmd=golden(x, p, f)
        x = x - lmd*p
        fr=fr.append({'X1':x[0], 'X2': x[1], 'F(x)': f(x[0], x[1]), 
                      'A': np.around(A,2), 
                      'lambda*p': np.around(lmd*p, 3),
                      'H_inv': np.around(np.linalg.inv(Hessian(x, f)), 2)}, 
				ignore_index=True)
        
        W = df(x0[0], x0[1], f) - df(x[0], x[1], f)
        X = x-x0
        
        A = A - np.outer(X.T, X)/np.dot(W, X) - 
		np.dot(A, np.outer( W, W.T)).dot(A.T)/(np.dot(np.dot(A,W), W))
                    
    return fr
