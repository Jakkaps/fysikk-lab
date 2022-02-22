import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.interpolate import CubicSpline

g = 9.81



# Horisontal avstand mellom festepunktene er 0.200 m
h = 0.200
xfast=np.asarray([0,h,2*h,3*h,4*h,5*h,6*h,7*h])


# Vi begrenser starthÃ¸yden (og samtidig den maksimale hÃ¸yden) til
# Ã¥ ligge mellom 250 og 300 mm
ymax = 300
# yfast: tabell med 8 heltall mellom 50 og 300 (mm); representerer
# Omregning fra mm til m:
# xfast = xfast/1000
# yfast = yfast/1000

# NÃ¥r programmet her har avsluttet while-lÃ¸kka, betyr det at
# tallverdiene i tabellen yfast vil resultere i en tilfredsstillende bane. 

#Programmet beregner deretter de 7 tredjegradspolynomene, et
#for hvert intervall mellom to nabofestepunkter.


#Med scipy.interpolate-funksjonen CubicSpline:
yfast = [0.296,  0.22, 0.17,  0.107, 0.119, 0.068, 0.015, 0.048]
cs = CubicSpline(xfast, yfast, bc_type='natural')

xmin = 0.000
xmax = 1.401
dx = 0.001

x = np.arange(xmin, xmax, dx)   

#funksjonen arange returnerer verdier paa det "halvaapne" intervallet
#[xmin,xmax), dvs slik at xmin er med mens xmax ikke er med. Her blir
#dermed x[0]=xmin=0.000, x[1]=xmin+1*dx=0.001, ..., x[1400]=xmax-dx=1.400, 
#dvs x blir en tabell med 1401 elementer
Nx = len(x)
y = cs(x)       #y=tabell med 1401 verdier for y(x)
dy = cs(x,1)    #dy=tabell med 1401 verdier for y'(x)
d2y = cs(x,2)   #d2y=tabell med 1401 verdier for y''(x)

#Eksempel: Plotter banens form y(x)
baneform = plt.figure('y(x)',figsize=(12,6))
""" plt.plot(x,y,xfast,yfast,'*') """
plt.title('Banens form')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.ylim(0.0,0.40)
plt.grid()
""" plt.show() """
#Figurer kan lagres i det formatet du foretrekker:
#baneform.savefig("baneform.pdf", bbox_inches='tight')
#baneform.savefig("baneform.png", bbox_inches='tight')
#baneform.savefig("baneform.eps", bbox_inches='tight')

fart = np.sqrt((10*g*(y[0] - y))/7)
fart_figur = plt.figure('y(x)',figsize=(12,6))
""" plt.plot(x, fart,'*') """
plt.title('Fart')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
""" plt.show() """

helningsvinkel = np.arctan(dy)
helningsvinkel_figur = plt.figure('y(x)',figsize=(12,6))
""" plt.plot(x, helningsvinkel,'*') """
plt.title('Helningsvinkel')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
""" plt.show() """

krumningsbanen = d2y / (1 + dy**2)**(3/2)
krumningsbane_figur = plt.figure('y(x)',figsize=(12,6))
""" plt.plot(x, krumningsbanen,'*') """
plt.title('Krumningsbane')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
""" plt.show() """

normalkraft = np.cos(helningsvinkel) + ((fart**2)*krumningsbanen)/ g
normalkraft_figur = plt.figure('y(x)',figsize=(12,6))
plt.plot(x, normalkraft,'-')
plt.title('Normalkraft')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
""" plt.show() """


tid = np.array((1,))
for i in range(1, len(x)):
	dt = (2*x[i]) / (fart[i-1]*np.cos(helningsvinkel[i-1]) + fart[i]*np.cos(helningsvinkel[i]))
	tid = np.append(tid, dt + tid[-1])

tid_figur = plt.figure('y(x)',figsize=(12,6))
""" plt.plot(tid, x,'-') """
plt.title('Tid')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
""" plt.show() """

m = 0.031
friksjon = np.absolute((m*g*2*np.sin(helningsvinkel)/7))
friksjon_normal = friksjon / (normalkraft*m*g)
friksjon_figur = plt.figure('y(x)',figsize=(12,6))
plt.plot(x, friksjon_normal,'-')
plt.title('Friksjonskraft')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
plt.show()

print(fart)
""" plt.plot(fart, tid,'-') """
plt.title('Fart')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
""" plt.show() """