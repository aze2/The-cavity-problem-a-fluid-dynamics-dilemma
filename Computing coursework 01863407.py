#!/usr/bin/env python
# coding: utf-8

# In[3]:


# In this section I am importing all the libraries I will need
import numpy
import matplotlib.pyplot

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

# In this section I am setting the domain of solution and the discretised grid
kx = 41

ky = 41

kt = 100

kit = 50

d = 1

dx = 2 / (kx - 1)

dy = 2 / (ky - 1)

rho = 1

nu = .1

dt = .001

# In this section I am defining arrays I would need (if needed)

x = numpy.linspace(0, 2, kx)

y = numpy.linspace(0, 2, ky)

X, Y = numpy.meshgrid(x, y)

u = numpy.zeros((ky, kx))
v = numpy.zeros((ky, kx))
b = numpy.zeros((ky, kx)) 
a = numpy.zeros((ky, kx))

# In this section I am setting the boundary conditions/initial values
def build_up_a(a, rho, dt, u, v, dx, dy):
    
    a[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return a
def pressure_poisson(b, dx, dy, a):
    
    bn = numpy.empty_like(b)
    
    bn = b.copy()
    
    for q in range(kit):
        bn = b.copy()
        b[1:-1, 1:-1] = (((bn[1:-1, 2:] + bn[1:-1, 0:-2]) * dy**2 + 
                          (bn[2:, 1:-1] + bn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          a[1:-1,1:-1])

        b[:, -1] = b[:, -2] # dp/dx = 0 at x = 2
        
        b[0, :] = b[1, :]   # dp/dy = 0 at y = 0
        
        b[:, 0] = b[:, 1]   # dp/dx = 0 at x = 0
        
        b[-1, :] = 0        # p = 0 at y = 2
        
    return b
def cavity_flow(kt, u, v, dt, dx, dy, b, rho, nu):
    un = numpy.empty_like(u)
    
    vn = numpy.empty_like(v)
    
    a = numpy.zeros((ky, kx))
    
    for n in range(kt):
        un = u.copy()
        vn = v.copy()
        
        a = build_up_a(a, rho, dt, u, v, dx, dy)
        b = pressure_poisson(b, dx, dy, a)
        
# In this section I am implementing the numerical method
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (b[1:-1, 2:] - b[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (b[2:, 1:-1] - b[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
        
    return u, v, b
u = numpy.zeros((ky, kx))
v = numpy.zeros((ky, kx))
b = numpy.zeros((ky, kx))
a = numpy.zeros((ky, kx))
kt = 100

u, v, b = cavity_flow(kt, u, v, dt, dx, dy, b, rho, nu)
fig = pyplot.figure(figsize=(11,7), dpi=100)


# plotting the pressure field as a contour
pyplot.contourf(X, Y, b, alpha=0.5, cmap=cm.viridis)  

pyplot.colorbar()

# plotting the pressure field outlines
pyplot.contour(X, Y, b, cmap=cm.viridis)  

# plotting velocity field
pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 

pyplot.xlabel('X')

pyplot.ylabel('Y');

fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, b, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, b, cmap=cm.viridis)
pyplot.streamplot(X, Y, u, v)
pyplot.xlabel('X')
pyplot.ylabel('Y');


# In[ ]:




