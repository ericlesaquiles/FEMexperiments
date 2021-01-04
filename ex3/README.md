# 2nd Order Problems
Consider the computational domain $\Omega = [0,L] \times [0,B]$ and the Diffusion-Convection equation with ho-mogeneous right hand side:

$$
\newcommand{\part}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\vec}[2]{\begin{pmatrix} #1\\ #2 \end{pmatrix}}
\begin{split}
-\nabla \cdot (K\nabla u ) + \beta \cdot \nabla u & = 0\text{, in $\Omega$} \\
\part{u}{y}(x,0) & = 0 \text{, x } \in [0,L] \\
\part{u}{y}(x,B) & = 0 \text{, x } \in [0,L] \\
u(0,y) & = 0 \text{, y } \in [0,B] \\
u(L,y) & = 1 \text{, y } \in [0,B]
\end{split}
$$

The parameter $K$ is the scalar diffusion coefficient and $\beta = \vec{\beta_1}{\beta_2}$ the convection velocity field.

* **(1) Continous variational formulation**

The continuous variational formulation is: "find u $\in H_{Dg}^{1}(\Omega)$ such that $\int_{\Omega}(\nabla v\cdot (K\nabla u)+v\beta\cdot u)\, d\Omega = a(u,v) = l(v) = \int_{\Omega} f v\, d\Omega$ for all $v \in H_{D0}^{1}$."

Observation: the general term of the linear form of second order ellipitic problems is:

$$
l(v) = \int_{\Omega} f v d\Omega + \int_{\Gamma}(v \textbf{n}\cdot (K\nabla u))d\Gamma
$$

However, we have that $\int_{\partial\Omega}(v \textbf{n}\cdot (K\nabla u))d\Gamma$ is null because on $\Gamma_{N}$ it values 0 as a boundary condition, and on $\Gamma_{D}$ we put 0 because the test function is 0 on Dirichlet boundary.

* **(2) Discrete variational formulation**

The discrete variational formulation is: "find u $\in P_{kg}(\Omega)$ such that $\int_{\Omega}(\nabla v\cdot (K\nabla u)+v\beta\cdot u) d\Omega = a(u,v) = l(v) = \int_{\Omega} f v \, d\Omega$ for all $v \in P_{k0}$."

$P_{kg} = \{v \in P_k, v = g \text{ on } \Gamma_{D}\}$, and $P_k \in H^1$.

In our case, we have $f = 0$.


```python
## Importing libraries

from fenics import *
from mshr import *
import numpy
from datetime import datetime
import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
```

### User defined parameters
#### Set the following parameters according to your problem


```python
def advection_diffusion(beta, 
                        k,
                        order,
                        mesh_refinement,
                        space_dim = 2,
                        L = 1.0,
                        B = 1.0,
                        r_border_refinements = 0,
                        stabilization_term = False,
                        fileio = 'pvd',
                        dir_ = './results'):
    #fileio = 'pvd'
    #dir_ = './results'

    if not os.path.exists(dir_):
        os.mkdir(dir_)

    if(space_dim == 2):
       domain = Rectangle(Point(0.0,0.0), Point(L,B))
    else:
       sys.exit("space_dim.eq.3 not implemented")
       
    # Calculo da camada limite para o refinamento
    mod_beta = np.sqrt(beta[0]^2+beta[1]^2)
    L_hcamadalimite = L - k/mod_beta

    # Thermal conductivity
    #k = 1.0
    kappa = Constant(k)

    # Convection velocity field
    beta = Constant(beta)

    # Order and refinement
    #order = 3
    #mesh_refinement = 40

    #### IO setup
    Pk = FiniteElement("Lagrange", 'triangle', order)

    ufile_pvd  = File(dir_+f'/temperatureO{order}mr{mesh_refinement}br{r_border_refinements}.pvd')
    domfile_pvd = File(dir_+f"/auxfuncO{order}mr{mesh_refinement}br{r_border_refinements}.pvd")

    # Order of finite elements space
    startTime = datetime.now()
    print(f'\n   ::> Begin computations for order {order} and refinement {mesh_refinement}')

    #### Mesh generation
    mesh = generate_mesh(domain, mesh_refinement)
    
        
    ### Dirichlet boundary conditions
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[0],0))
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[0],L))

    left    = Left()
    right   = Right()
    
    # Number of refinements
    # if r_border_refinements > 0:
    #     for i in range(r_border_refinements):
    #         edge_markers = MeshFunction('bool', mesh, mesh.topology().dim()-1)
    #         right.mark(edge_markers, True)

    #         mesh = refine(mesh, edge_markers)
            #mesh = mesh.child()
            
    if r_border_refinements > 0:
      for i in range(r_border_refinements):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in cells(mesh):
          vts = cell.get_vertex_coordinates()
          max_x = max(vts[0],vts[2],vts[4])
          if(max_x>=(L_hcamadalimite)):
            cell_markers[cell] = True
        mesh = refine(mesh, cell_markers)
    
    print("    |-Mesh done")
    print("    |--Number of vertices = "+str(mesh.num_vertices()))
    print("    |--Number of cells = "+str(mesh.num_cells()))
    print("    |--Cell size hmax,hmin = %.3g %.3g" % (mesh.hmax(), mesh.hmin()))

    
    # Finite element space
    W = FunctionSpace(mesh, Pk)
    print("    |--Total number of unknowns = %d" % (W.dim()))
    
    # Stabilization term (tk)
    if stabilization_term == True:
        DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
        VDG = FunctionSpace(mesh,DG)
        tk = Function(VDG)
        for cell in cells(mesh):
          tk.vector()[cell.index()] = 1/(4*kappa/mesh.hmax()**2 + 2*mod_beta/mesh.hmax())
    
    ### Variational formulation: Poisson problem
    u = TrialFunction(W)
    v = TestFunction(W)

    funcdom = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
    dx = Measure("dx")(subdomain_data=funcdom)

    f = Constant(0)
    if stabilization_term == False:
      a = inner(kappa*grad(u), grad(v))*dx + v*inner(beta,grad(u))*dx
    else:
      a = inner(kappa*grad(u), grad(v))*dx + v*inner(beta,grad(u))*dx + \
          (inner(beta,grad(u)) * tk *(inner(-1,kappa*grad(u))+inner(beta,grad(u))))*dx 

    LL = f*v*dx


    bcleft  = DirichletBC(W, Constant(0), left)
    bcright = DirichletBC(W, Constant(1), right)

    
    ### Solution
    w = Function(W)

    # solver.solve()
    solve(a == LL, w, [bcleft, bcright])

    ####### IO
    ufile_pvd << w
    domfile_pvd << funcdom
    
    #plt.figure(figsize=(10,10))
    #plot(mesh,linewidth=.1)
    #c = plot(w, wireframe=True, title=f'Solution')
    #plt.colorbar(c)
    #plt.show()
    
    u_exact = Expression("(exp(mod_beta*x[0]/kappa)-1)/(exp(mod_beta*L/kappa)-1)", degree = 3, kappa = kappa, beta = beta, mod_beta = mod_beta, L = L)
    

    return w, mesh
```

### Solutions for $\kappa \in \{10,1,.1,.01,.001\}$


```python
ks = [10, 1, .1, .01, .001]
solutions = [advection_diffusion(beta = (1,0), k = k, order = 3, mesh_refinement = 160, L = 2) 
             for k in ks]

for ((w, mesh), k) in zip(solutions, ks):
    plt.figure(figsize=(20,20))
    #plot(mesh,linewidth=.1)
    c = plot(w, wireframe=True, title=f"Solution for k = {k}")
    #plt.colorbar(c)
    plt.show()
```

    
       ::> Begin computations for order 3 and refinement 160
        |-Mesh done
        |--Number of vertices = 32099
        |--Number of cells = 63428
        |--Cell size hmax,hmin = 0.0126 0.00637
        |--Total number of unknowns = 286579
    
       ::> Begin computations for order 3 and refinement 160
        |-Mesh done
        |--Number of vertices = 32099
        |--Number of cells = 63428
        |--Cell size hmax,hmin = 0.0126 0.00637
        |--Total number of unknowns = 286579
    
       ::> Begin computations for order 3 and refinement 160
        |-Mesh done
        |--Number of vertices = 32099
        |--Number of cells = 63428
        |--Cell size hmax,hmin = 0.0126 0.00637
        |--Total number of unknowns = 286579
    
       ::> Begin computations for order 3 and refinement 160
        |-Mesh done
        |--Number of vertices = 32099
        |--Number of cells = 63428
        |--Cell size hmax,hmin = 0.0126 0.00637
        |--Total number of unknowns = 286579
    
       ::> Begin computations for order 3 and refinement 160
        |-Mesh done
        |--Number of vertices = 32099
        |--Number of cells = 63428
        |--Cell size hmax,hmin = 0.0126 0.00637
        |--Total number of unknowns = 286579



![png](output_6_1.png)



![png](output_6_2.png)



![png](output_6_3.png)



![png](output_6_4.png)



![png](output_6_5.png)


### Uniform mesh refinements for $\kappa = .01$


```python
k = .01
refinements = [40, 80, 160, 320]

solutions = [advection_diffusion(beta = (1,0), k = k, order = 3, mesh_refinement = mesh_refinement, L = 2) 
             for mesh_refinement in refinements]

for ((w, mesh), mr) in zip(solutions, refinements):
    plt.figure(figsize=(20,20))
    #plot(mesh,linewidth=.1)
    c = plot(w, wireframe=True, title=f'Solution for k = {k}, refinement = {mr}')
    #plt.colorbar(c)
    plt.show()
```

    
       ::> Begin computations for order 3 and refinement 40
        |-Mesh done
        |--Number of vertices = 2063
        |--Number of cells = 3932
        |--Cell size hmax,hmin = 0.0503 0.0262
        |--Total number of unknowns = 17983
    
       ::> Begin computations for order 3 and refinement 80
        |-Mesh done
        |--Number of vertices = 8131
        |--Number of cells = 15876
        |--Cell size hmax,hmin = 0.0252 0.0129
        |--Total number of unknowns = 72019
    
       ::> Begin computations for order 3 and refinement 160
        |-Mesh done
        |--Number of vertices = 32099
        |--Number of cells = 63428
        |--Cell size hmax,hmin = 0.0126 0.00637
        |--Total number of unknowns = 286579
    
       ::> Begin computations for order 3 and refinement 320
        |-Mesh done
        |--Number of vertices = 127460
        |--Number of cells = 253382
        |--Cell size hmax,hmin = 0.00629 0.00319
        |--Total number of unknowns = 1142524



![png](output_8_1.png)



![png](output_8_2.png)



![png](output_8_3.png)



![png](output_8_4.png)


### Local mesh refinement on rightmost boundary, takin $\kappa = .01$
First, so we have an idea on where the local refinement is happening, using global refinement of 40 and 4 local refinements


```python
k = .01
(w, mesh) = advection_diffusion(beta = (1,0), 
                                k = k, 
                                order = 3, 
                                mesh_refinement = 40, 
                                L = 2, 
                                r_border_refinements = 4) 

plt.figure(figsize=(20,20))
plot(mesh,linewidth=.3)
c = plot(w, wireframe=True, title=f'Solution for k = {k}, global refinement = {40}, local refinement = {4}')
#plt.colorbar(c)
plt.show()

plt.figure(figsize=(20,20))
#plot(mesh,linewidth=.3)
c = plot(w, wireframe=True, title=f'Solution for k = {k}, global refinement = {40}, local refinement = {4}')
#plt.colorbar(c)
plt.show()
```

    
       ::> Begin computations for order 3 and refinement 40
        |-Mesh done
        |--Number of vertices = 6379
        |--Number of cells = 12067
        |--Cell size hmax,hmin = 0.0503 0.00195
        |--Total number of unknowns = 55336



![png](output_10_1.png)



![png](output_10_2.png)



```python
k = .01
(w, mesh) = advection_diffusion(beta = (1,0), 
                                k = k, 
                                order = 3, 
                                mesh_refinement = 80, 
                                L = 2, 
                                r_border_refinements = 4) 

plt.figure(figsize=(20,20))
#plot(mesh,linewidth=.1)
c = plot(w, wireframe=True, title=f'Solution for k = {k}, global refinement = {80}, local refinement = {4}')
#plt.colorbar(c)
plt.show()
```

    
       ::> Begin computations for order 3 and refinement 80
        |-Mesh done
        |--Number of vertices = 22003
        |--Number of cells = 42631
        |--Cell size hmax,hmin = 0.0252 0.000888
        |--Total number of unknowns = 193900



![png](output_11_1.png)


### $\kappa = .001$, global mesh refinement = 160, local refinement = 4


```python
k = .001
r_border_refinements = 4

(w, mesh) = advection_diffusion(beta = (1,0), 
                                k = k, 
                                order = 3, 
                                mesh_refinement = 160, 
                                L = 2, 
                                r_border_refinements = r_border_refinements) 

plt.figure(figsize=(20,20))
#plot(mesh,linewidth=.1)
c = plot(w, wireframe=True, title=f'Solution for k = {k}, border refinement = {r_border_refinements}')
#plt.colorbar(c)
plt.show()
```

    
       ::> Begin computations for order 3 and refinement 160
        |-Mesh done
        |--Number of vertices = 43946
        |--Number of cells = 85190
        |--Cell size hmax,hmin = 0.0126 0.000476
        |--Total number of unknowns = 387406



![png](output_13_1.png)


## With stabilization term


```python
def advection_diffusion(beta, 
                        k,
                        order,
                        mesh_refinement,
                        space_dim = 2,
                        L = 1.0,
                        B = 1.0,
                        r_border_refinements = 0,
                        stabilization_term = False,
                        fileio = 'pvd',
                        dir_ = './results'):
    #fileio = 'pvd'
    #dir_ = './results'

    if not os.path.exists(dir_):
        os.mkdir(dir_)

    if(space_dim == 2):
       domain = Rectangle(Point(0.0,0.0), Point(L,B))
    else:
       sys.exit("space_dim.eq.3 not implemented")
       
    # Calculo da camada limite para o refinamento
    mod_beta = np.sqrt(beta[0]^2+beta[1]^2)
    L_hcamadalimite = L - k/mod_beta

    # Thermal conductivity
    #k = 1.0
    kappa = Constant(k)

    # Convection velocity field
    beta = Constant(beta)

    # Order and refinement
    #order = 3
    #mesh_refinement = 40

    #### IO setup
    Pk = FiniteElement("Lagrange", 'triangle', order)

    ufile_pvd  = File(dir_+f'/temperatureO{order}mr{mesh_refinement}br{r_border_refinements}.pvd')
    domfile_pvd = File(dir_+f"/auxfuncO{order}mr{mesh_refinement}br{r_border_refinements}.pvd")

    # Order of finite elements space
    startTime = datetime.now()
    print(f'\n   ::> Begin computations for order {order} and refinement {mesh_refinement}')

    #### Mesh generation
    mesh = generate_mesh(domain, mesh_refinement)
    
        
    ### Dirichlet boundary conditions
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[0],0))
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[0],L))

    left    = Left()
    right   = Right()
    
    # Number of refinements
    # if r_border_refinements > 0:
    #     for i in range(r_border_refinements):
    #         edge_markers = MeshFunction('bool', mesh, mesh.topology().dim()-1)
    #         right.mark(edge_markers, True)

    #         mesh = refine(mesh, edge_markers)
            #mesh = mesh.child()
            
    if r_border_refinements > 0:
      for i in range(r_border_refinements):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in cells(mesh):
          vts = cell.get_vertex_coordinates()
          max_x = max(vts[0],vts[2],vts[4])
          if(max_x>=(L_hcamadalimite)):
            cell_markers[cell] = True
        mesh = refine(mesh, cell_markers)
    
    print("    |-Mesh done")
    print("    |--Number of vertices = "+str(mesh.num_vertices()))
    print("    |--Number of cells = "+str(mesh.num_cells()))
    print("    |--Cell size hmax,hmin = %.3g %.3g" % (mesh.hmax(), mesh.hmin()))

    
    # Finite element space
    W = FunctionSpace(mesh, Pk)
    print("    |--Total number of unknowns = %d" % (W.dim()))
    
    # Stabilization term (tk)
    if stabilization_term:
      DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
      VDG = FunctionSpace(mesh,DG)
      tk = Function(VDG)
      for cell in cells(mesh):
        tk.vector()[cell.index()] = 1/(4*kappa/mesh.hmax()**2 + 2*mod_beta/mesh.hmax())
    
    ### Variational formulation: Poisson problem
    u = TrialFunction(W)
    v = TestFunction(W)

    funcdom = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
    dx = Measure("dx")(subdomain_data=funcdom)

    f = Constant(0)
    if not stabilization_term:
      a = inner(kappa*grad(u), grad(v))*dx + v*inner(beta,grad(u))*dx
    else:
      a = inner(kappa*grad(u), grad(v))*dx + v*inner(beta,grad(u))*dx + \
          inner(beta,grad(u)) * tk *(inner(-1,kappa*grad(u))+inner(beta,grad(u)))*dx 

    LL = f*v*dx


    bcleft  = DirichletBC(W, Constant(0), left)
    bcright = DirichletBC(W, Constant(1), right)

    
    ### Solution
    w = Function(W)

    # solver.solve()
    solve(a == LL, w, [bcleft, bcright])

    ####### IO
    ufile_pvd << w
    domfile_pvd << funcdom
    
    #plt.figure(figsize=(10,10))
    #plot(mesh,linewidth=.1)
    #c = plot(w, wireframe=True, title=f'Solution')
    #plt.colorbar(c)
    #plt.show()
    
    u_exact = Expression("(exp(mod_beta*x[0]/kappa)-1)/(exp(mod_beta*L/kappa)-1)", 
                         degree = 3, kappa = kappa, 
                         beta = beta, 
                         mod_beta = mod_beta, 
                         L = L)
    error_L2 = errornorm(u_exact,w,'L2')
    error_H1 = errornorm(u_exact,w,'H1')
    print('|  error L2 = ' + str(error_L2))
    print('|  error H1 = ' + str(error_H1))

    return w, mesh
```


```python
k = .001
#r_border_refinements = 2

(w, mesh) = advection_diffusion(beta = (1,0), 
                                k = k, 
                                order = 3, 
                                mesh_refinement = 80, 
                                L = 2, 
                                stabilization_term = True) 

plt.figure(figsize=(20,20))
#plot(mesh,linewidth=.1)
c = plot(w, wireframe=True, title=f'Solution for k = {k}')
#plt.colorbar(c)
plt.show()
```

    
       ::> Begin computations for order 3 and refinement 80
        |-Mesh done
        |--Number of vertices = 8131
        |--Number of cells = 15876
        |--Cell size hmax,hmin = 0.0252 0.0129
        |--Total number of unknowns = 72019
    Shapes do not match: <IntValue id=140544876089672> and <ComponentTensor id=140544875317912>.



    ---------------------------------------------------------------------------

    UFLException                              Traceback (most recent call last)

    <ipython-input-19-23bd36308ed6> in <module>
          7                                 mesh_refinement = 80,
          8                                 L = 2,
    ----> 9                                 stabilization_term = True) 
         10 
         11 plt.figure(figsize=(20,20))


    <ipython-input-18-f21b1e0ed587> in advection_diffusion(beta, k, order, mesh_refinement, space_dim, L, B, r_border_refinements, stabilization_term, fileio, dir_)
        111     else:
        112       a = inner(kappa*grad(u), grad(v))*dx + v*inner(beta,grad(u))*dx + \
    --> 113           inner(beta,grad(u)) * tk *(inner(-1,kappa*grad(u))+inner(beta,grad(u)))*dx
        114 
        115     LL = f*v*dx


    /usr/lib/python3/dist-packages/ufl/operators.py in inner(a, b)
        156     if a.ufl_shape == () and b.ufl_shape == ():
        157         return a * Conj(b)
    --> 158     return Inner(a, b)
        159 
        160 


    /usr/lib/python3/dist-packages/ufl/tensoralgebra.py in __new__(cls, a, b)
        145         ash, bsh = a.ufl_shape, b.ufl_shape
        146         if ash != bsh:
    --> 147             error("Shapes do not match: %s and %s." % (ufl_err_str(a), ufl_err_str(b)))
        148 
        149         # Simplification


    /usr/lib/python3/dist-packages/ufl/log.py in error(self, *message)
        156         "Write error message and raise an exception."
        157         self._log.error(*message)
    --> 158         raise self._exception_type(self._format_raw(*message))
        159 
        160     def begin(self, *message):


    UFLException: Shapes do not match: <IntValue id=140544876089672> and <ComponentTensor id=140544875317912>.

