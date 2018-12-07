import numpy as np
import matplotlib.pyplot as plt
import sys

stds = []
means =[]
TodosDatos=[]
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
#Sacamos los datos creados en C
for i in range(8):
    datos = np.genfromtxt("sample_%d.dat" % i)
    TodosDatos.append(datos)
    ax1.hist(datos,bins=100,alpha=0.3, density=True)

#la estadistica Gelman-Rubin en funcion del numero de iteraciones
#for j in range(2,1000):
#    (j-1)/(j*8)*np.sum(np.std(TodosDatos[:,:j],axis=0)**2,axis=1) #+ (9)/(8*(7))*np.sum(np.std(TodosDatos[:,:j],axis=0)**2,axis=1)

x = np.linspace(-5,5,1000)
y = np.exp(-x**2/(2))/(np.sqrt(2*np.pi)*1)
ax1.plot(x,y)
ax1.set_title("Histogramas de las funciones de probabilidad generadas")
fig.savefig('histograma.png')