import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pylab
import copy

def fA(x):
    return (-20 * np.e ** (-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) -
            np.e ** (0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) +
            np.e + 20)

def fMulti(x):
    return x[0]**2 + x[1]**2 - np.cos(18 * x[0]) - np.cos(18 * x[1])

def fQuad(x):
    return 10 * x[0]**2 - 4 * x[0] * x[1] + 7 * x[1]**2 - 4 * np.sqrt(5) * (5 * x[0] + x[1]) - 16


def create_crom(quan, leng, dim):
    population = [np.random.choice([0, 1], size=(dim, leng)) for x in range (quan)]
    return population

def convert(crom, area):
    #print(crom)
    x = []
    for i in range (0,len(crom)):
        summ=0
        for k in range (len(crom[i])):
            summ+=crom[i][k]*2**k 
        x.append(summ)
        x[i]=area[i][0]+x[i]*((area[i][1]-area[i][0])/(2**len(crom[i])-1))
    return x

def fitness(dec_crom,function):
    return function(dec_crom)

def select(t, population,area,func):
    final=[]
    for i in range(len(population)):
        x=random.choices(population,k=t)
        x_fit=[fitness(convert(i,area),func) for i in x]
        num=x_fit.index(min(x_fit))
        final.append(copy.copy(x[num]))
    return final

def cros(parents):
    #print("parents",parents)
    child =[]
    x=[]
    y=list(zip(*[iter(parents)] * 2))
    #print(y[0][0][0][:2][0])
    for i in range(len(y)):
        stop=random.randint(1,len(y[0][0][0])-1)
        for k in range(2):
                #print(y[i][k][m][:stop])
                #print(y[i][int(not(k))][m][stop:])
            d=[(list((y[i][k][m][:stop]))+list((y[i][int(not(k))][m][stop:]))) for m in range(2)]
            child.append(d)
            #print("child",child)
    return child
          
def mutate(population, prob):
    #print(population[0][0])
    num = int(prob*len(list(population[0][0])))
    for s in range(len(population)):
        for n in range (0,len(population[s])):
            el = np.random.choice(range(0,len(population[s][n])),num, replace=False)
            #print("start",population[s][n])
            for p in el:
                population[s][n][p]=int(not(population[s][n][p]))
            #print("end",population[s][n])
    return population
def genetic(func, quan, leng, dim, area, gen,t,prob):
    population=create_crom(quan, leng, dim)
    for number in range(0,gen):
        con_crom=[]
        for i in range(quan):
            con_crom.append(convert(population[i],area))
            for k in range(dim):
                xdot[k].append(con_crom[i][k])
        fit=[fitness(con_crom[i],func) for i in range(quan)]
        xdot[2].extend(fit)
        #zdot.extend([-1*i for i in fit])
        parents=select(t,population,area,func)
        #print("before",population)
        population=mutate(cros(parents),prob)
        #print("after",population)
    fit=[fitness(convert(i,area),func) for i in population]
    #print(fit)
    ans=population[fit.index(min(fit))]
    print(fitness(convert(ans, area),func))
    return convert(ans, area)

xdot=[]
[xdot.append([]) for i in range(3)]
func = fQuad
genetic(func, 100, 22, 2, [[-1,1],[-1,1]], 70, 5, 0.05)

def makeData ():
    x = np.arange (-1, 1, 0.01)
    y = np.arange (-1, 1, 0.01)
    xgrid, ygrid = np.meshgrid(x, y)

    zgrid = func([xgrid,ygrid])
    return xgrid, ygrid, zgrid

x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)

axes.scatter(xdot[0],xdot[1],xdot[2],c='b',s=20)
axes.plot_surface(x, y, z,color='c', alpha = 0.3)

fig.set_figwidth(14)
fig.set_figheight(14)

pylab.show()