# Error, interpolation and projection
Consider the Poisson problem on the domain $\Omega = [0,1] \times [0,1]$, source term f and thermal diffusivity $\kappa$. Find U satisfying


$$
\begin{split}
-\nabla \cdot (\kappa(x) \nabla u(x)) & = f(x), x \in \Omega\\
u(x) & = cos(2\pi x_1)cos(2 \pi x_2)
\end{split}
$$

Making $\kappa = 1$, $f = 8\pi^2 cos(2\pi x_1)cos(2\pi x_2)$, the exact solution reads

$$
u(x) = cos(2\pi x_1)cos(2\pi x_2)
$$

The solution $u_h$ is sought in a finite dimensional space $V_h \subset H^1(\Omega)$, made up of polynomial functions of degree k. In the variational formulation one looks for $u_h in V_h$ such that
$$
\int_{\Omega} \nabla u_h \cdot \nabla v_h dx \int_{\Omega} f v_h dx
$$

In the following, the problem is solved, in different settings (with different mesh refinements and different $V_h$s), and then it is computed the errors L2 and H1 and the runtime of each running, as an effort to develop a way to determine which setting to use in a situation.


```python
## Importing libraries

from dolfin import *
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
space_dim = 2

Lx = 1.0
Ly = 1.0

fileio = 'pvd'
dir_ = './results'

if not os.path.exists(dir_):
    os.mkdir(dir_)

if(space_dim == 2):
   domain = Rectangle(Point(0.0,0.0), Point(Lx,Ly))
else:
   sys.exit("space_dim.eq.3 not implemented")

# Thermal conductivity
k = 1.0
kappa = Constant(k)

# In order to do time averaging
times = 3 
```

### Set the problem and solve it, recording relevant info


```python
# Order of finite elements space
orders = [1,2,3]
mesh_refinements = [20,40,80,160]
elapsed_time = []

for t in range(times):
    # For remembering relevant info
    elapsed_time.append([])

    errorsH1        = []
    errorsL2        = []
    errors_interpL2 = []
    errors_interpH1 = []
    unknowns        = []
    hmax            = []

    for io, order in enumerate(orders):
        elapsed_time[t].append([])
        errorsH1.append([])
        errorsL2.append([])
        errors_interpH1.append([])
        errors_interpL2.append([])
        unknowns.append([])
        hmax.append([])
        for im, mesh_refinement in enumerate(mesh_refinements):

            Pk = FiniteElement("Lagrange", 'triangle', order)

            #### IO setup
            ufile_pvd  = File(dir_+f'/temperatureO{order}mr{mesh_refinement}.pvd')
            domfile_pvd = File(dir_+f"/auxfuncO{order}mr{mesh_refinement}.pvd")

            startTime = datetime.now()
            print(f'\n   ::> Begin computations for order {order} and refinement {mesh_refinement} at {t}')


            #### Mesh generation
            mesh = generate_mesh(domain, mesh_refinement)
            mesh = UnitSquareMesh.create(mesh_refinement, 
                                     mesh_refinement, 
                                     CellType.Type.triangle)

            #print("    |-Mesh done")
            #print("    |--Number of vertices = "+str(mesh.num_vertices()))
            #print("    |--Number of cells = "+str(mesh.num_cells()))
            #print("    |--Cell size hmax,hmin = %.3g %.3g" % (mesh.hmax(), mesh.hmin()))


            # Finite element space
            W = FunctionSpace(mesh, Pk)
            print("    |--Total number of unknowns = %d" % (W.dim()))
            if t == 1:
                unknowns[io].append(W.dim())

            ### Variational formulation: Poisson problem
            u = TrialFunction(W)
            v = TestFunction(W)

            funcdom = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
            dx = Measure("dx")(subdomain_data=funcdom)
            #dx = Measure("dx")

            # Volumetric source
            f = Expression("8*pow(pi,2)*cos(2*pi*x[0])*cos(2*pi*x[1])", degree = 5)

            a = inner(kappa*grad(u), grad(v))*dx
            L = f*v*dx

            #### Dirichlet boundary conditions on W
            Tboundary = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])", degree = 5)
            bcs = DirichletBC(W, Tboundary, DomainBoundary())


            ### Solution
            w = Function(W)

            # For Mesh refinement
            # problem = LinearVariationalProblem(a, L, w, bcs)
            # solver = LinearVariationalSolver(problem)
            # prm = solver.parameters
            # linear_solver = 'Krylov'
            # if linear_solver == 'Krylov':
            #     prm ["linear_solver"]= "gmres"
            #     prm ["preconditioner"]= "ilu"
            #     prm ["krylov_solver"]["absolute_tolerance"]= 1e-11
            #     prm ["krylov_solver"]["relative_tolerance"]= 1e-10
            #     prm ["krylov_solver"]["maximum_iterations"]= 10000
            #     prm ["krylov_solver"]["monitor_convergence"]= True
            #     prm ["krylov_solver"]["nonzero_initial_guess"]= True
            # else :
            #     prm ["linear_solver"]= "mumps"

            # solver.solve()
            solve(a == L, w, bcs)
            #      solver_parameters={'linear_solver':'gmres', 'preconditioner':'ilu'})

            ####### IO
            ufile_pvd << w
            domfile_pvd << funcdom

            if t == 1:
                elapsed_time[t][io].append(datetime.now() - startTime)
            #print('\n    ::> Elapsed time: ', str(elapsed_time[io][im]), '\n')


            # plt.figure(figsize=(15,15))
            # plot(mesh,linewidth=.2)
            # c = plot(w, wireframe=True, title=f'Solution')
            # plt.colorbar(c)
            # plt.show()
            if t == 1:
                hmax[io].append(mesh.hmax())

            #### Compute errors
            uexact = Expression('cos(2*pi*x[0])*cos(2*pi*x[1])', degree = 7)
            errorsH1[io].append(errornorm(uexact, w, 'H1'))
            errorsL2[io].append(errornorm(uexact, w, 'L2'))


            # Interpolate exact solution and compute interpolation error
            W = FunctionSpace (mesh, Pk)

            uinter = interpolate(uexact , W)
            errors_interpH1[io].append(errornorm(uexact, w, 'H1'))
            errors_interpL2[io].append(errornorm(uexact, w, 'L2'))
```

    
       ::> Begin computations for order 1 and refinement 20 at 0
        |--Total number of unknowns = 441
    
       ::> Begin computations for order 1 and refinement 40 at 0
        |--Total number of unknowns = 1681
    
       ::> Begin computations for order 1 and refinement 80 at 0
        |--Total number of unknowns = 6561
    
       ::> Begin computations for order 1 and refinement 160 at 0
        |--Total number of unknowns = 25921
    
       ::> Begin computations for order 2 and refinement 20 at 0
        |--Total number of unknowns = 1681
    
       ::> Begin computations for order 2 and refinement 40 at 0
        |--Total number of unknowns = 6561
    
       ::> Begin computations for order 2 and refinement 80 at 0
        |--Total number of unknowns = 25921
    
       ::> Begin computations for order 2 and refinement 160 at 0
        |--Total number of unknowns = 103041
    
       ::> Begin computations for order 3 and refinement 20 at 0
        |--Total number of unknowns = 3721
    
       ::> Begin computations for order 3 and refinement 40 at 0
        |--Total number of unknowns = 14641
    
       ::> Begin computations for order 3 and refinement 80 at 0
        |--Total number of unknowns = 58081
    
       ::> Begin computations for order 3 and refinement 160 at 0
        |--Total number of unknowns = 231361
    
       ::> Begin computations for order 1 and refinement 20 at 1
        |--Total number of unknowns = 441
    
       ::> Begin computations for order 1 and refinement 40 at 1
        |--Total number of unknowns = 1681
    
       ::> Begin computations for order 1 and refinement 80 at 1
        |--Total number of unknowns = 6561
    
       ::> Begin computations for order 1 and refinement 160 at 1
        |--Total number of unknowns = 25921
    
       ::> Begin computations for order 2 and refinement 20 at 1
        |--Total number of unknowns = 1681
    
       ::> Begin computations for order 2 and refinement 40 at 1
        |--Total number of unknowns = 6561
    
       ::> Begin computations for order 2 and refinement 80 at 1
        |--Total number of unknowns = 25921
    
       ::> Begin computations for order 2 and refinement 160 at 1
        |--Total number of unknowns = 103041
    
       ::> Begin computations for order 3 and refinement 20 at 1
        |--Total number of unknowns = 3721
    
       ::> Begin computations for order 3 and refinement 40 at 1
        |--Total number of unknowns = 14641
    
       ::> Begin computations for order 3 and refinement 80 at 1
        |--Total number of unknowns = 58081
    
       ::> Begin computations for order 3 and refinement 160 at 1
        |--Total number of unknowns = 231361
    
       ::> Begin computations for order 1 and refinement 20 at 2
        |--Total number of unknowns = 441
    
       ::> Begin computations for order 1 and refinement 40 at 2
        |--Total number of unknowns = 1681
    
       ::> Begin computations for order 1 and refinement 80 at 2
        |--Total number of unknowns = 6561
    
       ::> Begin computations for order 1 and refinement 160 at 2
        |--Total number of unknowns = 25921
    
       ::> Begin computations for order 2 and refinement 20 at 2
        |--Total number of unknowns = 1681
    
       ::> Begin computations for order 2 and refinement 40 at 2
        |--Total number of unknowns = 6561
    
       ::> Begin computations for order 2 and refinement 80 at 2
        |--Total number of unknowns = 25921
    
       ::> Begin computations for order 2 and refinement 160 at 2
        |--Total number of unknowns = 103041
    
       ::> Begin computations for order 3 and refinement 20 at 2
        |--Total number of unknowns = 3721
    
       ::> Begin computations for order 3 and refinement 40 at 2
        |--Total number of unknowns = 14641
    
       ::> Begin computations for order 3 and refinement 80 at 2
        |--Total number of unknowns = 58081
    
       ::> Begin computations for order 3 and refinement 160 at 2
        |--Total number of unknowns = 231361



```python
elapsed_time
```




    [[[], [], []],
     [[datetime.timedelta(0, 0, 31462),
       datetime.timedelta(0, 0, 266996),
       datetime.timedelta(0, 0, 535344),
       datetime.timedelta(0, 1, 527540)],
      [datetime.timedelta(0, 0, 70815),
       datetime.timedelta(0, 0, 284064),
       datetime.timedelta(0, 0, 948603),
       datetime.timedelta(0, 4, 22690)],
      [datetime.timedelta(0, 0, 84254),
       datetime.timedelta(0, 0, 348830),
       datetime.timedelta(0, 1, 925579),
       datetime.timedelta(0, 6, 883775)]],
     [[], [], []]]




```python
plt.figure(figsize=(10,10))
plot(mesh,linewidth=.2)
plot(w)
```




    <matplotlib.tri.tricontour.TriContourSet at 0x7f4b8c4b8f28>




![png](output_7_1.png)


### Framing the data


```python

```


```python
idx = [1,2,3] # for indexing the dataframes' rows


elapsed_time_df0 = pd.DataFrame(data = [[time.total_seconds() for time in el] for el in elapsed_time[0]],
                               columns = mesh_refinements,
                               index = idx)

elapsed_time_df1 = pd.DataFrame(data = [[time.total_seconds() for time in el] for el in elapsed_time[1]],
                               columns = mesh_refinements,
                               index = idx)

elapsed_time_df2 = pd.DataFrame(data = [[time.total_seconds() for time in el] for el in elapsed_time[2]],
                               columns = mesh_refinements,
                               index = idx)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _list_to_arrays(data, columns, coerce_float, dtype)
        496         result = _convert_object_array(
    --> 497             content, columns, dtype=dtype, coerce_float=coerce_float
        498         )


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _convert_object_array(content, columns, coerce_float, dtype)
        580             raise AssertionError(
    --> 581                 f"{len(columns)} columns passed, passed data had "
        582                 f"{len(content)} columns"


    AssertionError: 4 columns passed, passed data had 0 columns

    
    The above exception was the direct cause of the following exception:


    ValueError                                Traceback (most recent call last)

    <ipython-input-102-57d83ec236b6> in <module>
          4 elapsed_time_df0 = pd.DataFrame(data = [[time.total_seconds() for time in el] for el in elapsed_time[0]],
          5                                columns = mesh_refinements,
    ----> 6                                index = idx)
          7 
          8 elapsed_time_df1 = pd.DataFrame(data = [[time.total_seconds() for time in el] for el in elapsed_time[1]],


    ~/.local/lib/python3.6/site-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
        472                     if is_named_tuple(data[0]) and columns is None:
        473                         columns = data[0]._fields
    --> 474                     arrays, columns = to_arrays(data, columns, dtype=dtype)
        475                     columns = ensure_index(columns)
        476 


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in to_arrays(data, columns, coerce_float, dtype)
        459         return [], []  # columns if columns is not None else []
        460     if isinstance(data[0], (list, tuple)):
    --> 461         return _list_to_arrays(data, columns, coerce_float=coerce_float, dtype=dtype)
        462     elif isinstance(data[0], abc.Mapping):
        463         return _list_of_dict_to_arrays(


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _list_to_arrays(data, columns, coerce_float, dtype)
        498         )
        499     except AssertionError as e:
    --> 500         raise ValueError(e) from e
        501     return result
        502 


    ValueError: 4 columns passed, passed data had 0 columns



```python
elapsed_time_df2.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>20</th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.055476</td>
      <td>0.335549</td>
      <td>0.823233</td>
      <td>3.524624</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.031796</td>
      <td>0.259176</td>
      <td>0.613829</td>
      <td>2.621540</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.026605</td>
      <td>0.138238</td>
      <td>0.408401</td>
      <td>1.410543</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.038436</td>
      <td>0.188787</td>
      <td>0.470666</td>
      <td>2.057958</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.050268</td>
      <td>0.239337</td>
      <td>0.532932</td>
      <td>2.705373</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.069911</td>
      <td>0.434205</td>
      <td>1.030648</td>
      <td>4.581664</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.089554</td>
      <td>0.629073</td>
      <td>1.528365</td>
      <td>6.457955</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sns.set_theme(style="ticks", color_codes=True)

grid = sns.lineplot(data=elapsed_time_df0)
grid.set(xscale="linear", yscale="log")

grid = sns.lineplot(data=elapsed_time_df1)
grid.set(xscale="linear", yscale="log")

grid = sns.lineplot(data=elapsed_time_df2)
grid.set(xscale="linear", yscale="log")
grid.set(xticks = range(1,4))

fig = grid.get_figure()
fig.savefig("time_comparison.png")
```


![png](output_12_0.png)



```python
grid = sns.catplot(data=elapsed_time_df0)
grid.set(xscale="linear", yscale="log", xticks=range(1,4))

```




    <seaborn.axisgrid.FacetGrid at 0x7f4b8c7a2358>




![png](output_13_1.png)



```python
errorsH1_df  = pd.DataFrame(data = errorsH1,
                            columns = mesh_refinements,
                            index = [1,2,3])


errorsL2_df  = pd.DataFrame(data = errorsL2,
                            columns = mesh_refinements,
                            index = [1,2,3])


errors_interpL2_df  = pd.DataFrame(data = errors_interpL2,
                                   columns = mesh_refinements,
                                   index = [1,2,3])


errors_interpH1_df  = pd.DataFrame(data = errors_interpH1,
                                   columns = mesh_refinements,
                                   index = [1,2,3])


unknowns_df  = pd.DataFrame(data = unknowns,
                            columns = mesh_refinements,
                            index = [1,2,3])


hmax_df  = pd.DataFrame(data = hmax,
                        columns = mesh_refinements,
                        index = [1,2,3])

errorsH1_df
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _list_to_arrays(data, columns, coerce_float, dtype)
        496         result = _convert_object_array(
    --> 497             content, columns, dtype=dtype, coerce_float=coerce_float
        498         )


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _convert_object_array(content, columns, coerce_float, dtype)
        580             raise AssertionError(
    --> 581                 f"{len(columns)} columns passed, passed data had "
        582                 f"{len(content)} columns"


    AssertionError: 4 columns passed, passed data had 0 columns

    
    The above exception was the direct cause of the following exception:


    ValueError                                Traceback (most recent call last)

    <ipython-input-104-da86c7a92d61> in <module>
         21 unknowns_df  = pd.DataFrame(data = unknowns,
         22                             columns = mesh_refinements,
    ---> 23                             index = [1,2,3])
         24 
         25 


    ~/.local/lib/python3.6/site-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
        472                     if is_named_tuple(data[0]) and columns is None:
        473                         columns = data[0]._fields
    --> 474                     arrays, columns = to_arrays(data, columns, dtype=dtype)
        475                     columns = ensure_index(columns)
        476 


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in to_arrays(data, columns, coerce_float, dtype)
        459         return [], []  # columns if columns is not None else []
        460     if isinstance(data[0], (list, tuple)):
    --> 461         return _list_to_arrays(data, columns, coerce_float=coerce_float, dtype=dtype)
        462     elif isinstance(data[0], abc.Mapping):
        463         return _list_of_dict_to_arrays(


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _list_to_arrays(data, columns, coerce_float, dtype)
        498         )
        499     except AssertionError as e:
    --> 500         raise ValueError(e) from e
        501     return result
        502 


    ValueError: 4 columns passed, passed data had 0 columns



```python
grid = sns.lineplot(data=elapsed_time_df)
grid.set(xscale="linear", yscale="linear")
```




    [None, None]




![png](output_15_1.png)



```python
grid = sns.lineplot(data=errorsL2_df)
grid.set(xscale="linear", yscale="log")
```




    [None, None]




![png](output_16_1.png)



```python
errorsL2_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.007136</td>
      <td>1.673595e-03</td>
      <td>4.377572e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000128</td>
      <td>1.573341e-05</td>
      <td>1.963030e-06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000002</td>
      <td>1.378058e-07</td>
      <td>8.800856e-09</td>
    </tr>
  </tbody>
</table>
</div>




```python
elapsed_time_df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>20</th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.024095</td>
      <td>0.169513</td>
      <td>0.508197</td>
      <td>1.627948</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.080577</td>
      <td>0.243997</td>
      <td>0.900613</td>
      <td>3.203450</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.166473</td>
      <td>0.572006</td>
      <td>1.587737</td>
      <td>6.645786</td>
    </tr>
  </tbody>
</table>
</div>




```python
errors_interpL2_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.007136</td>
      <td>1.673595e-03</td>
      <td>4.377572e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000128</td>
      <td>1.573341e-05</td>
      <td>1.963030e-06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000002</td>
      <td>1.378058e-07</td>
      <td>8.800856e-09</td>
    </tr>
  </tbody>
</table>
</div>




```python
errors_interpH1_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.659578</td>
      <td>0.331560</td>
      <td>0.165164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.025271</td>
      <td>0.006312</td>
      <td>0.001575</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000633</td>
      <td>0.000078</td>
      <td>0.000010</td>
    </tr>
  </tbody>
</table>
</div>




```python
hmax_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.049998</td>
      <td>0.024998</td>
      <td>0.0125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.049998</td>
      <td>0.024998</td>
      <td>0.0125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.049998</td>
      <td>0.024998</td>
      <td>0.0125</td>
    </tr>
  </tbody>
</table>
</div>



### Solve with quadrilateral elements


```python
# For remembering relevant info
Qelapsed_time  = []
QerrorsH1      = []
QerrorsL2      = []
Qerrors_interp = []
Qunknowns      = []
Qhmax          = []

# Order of finite elements space
orders = [1,2,3]
mesh_refinements = [40,80,160]

for io, order in enumerate(orders):
    Qelapsed_time.append([])
    QerrorsH1.append([])
    QerrorsL2.append([])
    Qerrors_interp.append([])
    Qunknowns.append([])
    Qhmax.append([])
    
    for im, mesh_refinement in enumerate(mesh_refinements):
        
        Qk = FiniteElement("Lagrange", 'quadrilateral', order)

        #### IO setup
        ufile_pvd  = File(dir_+f'/QtemperatureO{order}mr{mesh_refinement}.pvd')
        domfile_pvd = File(dir_+f"/QauxfuncO{order}mr{mesh_refinement}.pvd")

        startTime = datetime.now()
        print(f'\n   ::> Begin computations for order {order} and refinement {mesh_refinement}')


        #### Mesh generation
        mesh = UnitSquareMesh.create(mesh_refinement, 
                                     mesh_refinement, 
                                     CellType.Type.quadrilateral)

        #print("    |-Mesh done")
        #print("    |--Number of vertices = "+str(mesh.num_vertices()))
        #print("    |--Number of cells = "+str(mesh.num_cells()))
        #print("    |--Cell size hmax,hmin = %.3g %.3g" % (mesh.hmax(), mesh.hmin()))


        # Finite element space
        W = FunctionSpace(mesh, Qk)
        #W = FunctionSpace(mesh, Qk)
        print("    |--Total number of unknowns = %d" % (W.dim()))
        Qunknowns[io].append(W.dim())
        
        ### Variational formulation: Poisson problem
        u = TrialFunction(W)
        v = TestFunction(W)

        funcdom = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
        dx = Measure("dx")(subdomain_data=funcdom)
        #dx = Measure("dx")

        # Volumetric source
        f = Expression("8*pow(pi,2)*cos(2*pi*x[0])*cos(2*pi*x[1])", degree = 5)

        a = inner(kappa*grad(u), grad(v))*dx
        L = f*v*dx

        #### Dirichlet boundary conditions on W
        Tboundary = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])", degree = 5)
        bcs = DirichletBC(W, Tboundary, DomainBoundary())


        ### Solution
        w = Function(W)

        # For Mesh refinement
        # problem = LinearVariationalProblem(a, L, w, bcs)
        # solver = LinearVariationalSolver(problem)
        # prm = solver.parameters
        # linear_solver = 'Krylov'
        # if linear_solver == 'Krylov':
        #     prm ["linear_solver"]= "gmres"
        #     prm ["preconditioner"]= "ilu"
        #     prm ["krylov_solver"]["absolute_tolerance"]= 1e-11
        #     prm ["krylov_solver"]["relative_tolerance"]= 1e-10
        #     prm ["krylov_solver"]["maximum_iterations"]= 10000
        #     prm ["krylov_solver"]["monitor_convergence"]= True
        #     prm ["krylov_solver"]["nonzero_initial_guess"]= True
        # else :
        #     prm ["linear_solver"]= "mumps"

        # solver.solve()
        solve(a == L, w, bcs)
        #      solver_parameters={'linear_solver':'gmres', 'preconditioner':'ilu'})

        ####### IO
        ufile_pvd << w
        domfile_pvd << funcdom

        Qelapsed_time[io].append(datetime.now() - startTime)
        print('\n  ::> Elapsed time: ', str(elapsed_time[io][im]), '\n')


        # plt.figure(figsize=(15,15))
        # plot(mesh,linewidth=.2)
        # c = plot(w, wireframe=True, title=f'Solution')
        # plt.colorbar(c)
        # plt.show()

        Qhmax[io].append(mesh.hmax())
        
        #### Compute errors
        uexact = Expression('cos(2*pi*x[0])*cos(2*pi*x[1])', degree = 7)
        gradue = Expression(('-2*pi*cos(2*pi*x[1])*sin(2*pi*x[0])','-2*pi*sin(2*pi*x[1])*cos(2*pi*x[0])'), 
                            degree = 7)
        erl2 = np.sqrt(assemble((uexact - w)**2*dx))
        erh1 = np.sqrt(assemble(((uexact - w)**2 + (grad(w) - gradue)**2 ) *dx))
        
        QerrorsH1[io].append(erh1)
        QerrorsL2[io].append(erl2)


        # Interpolate exact solution and compute interpolation error
        W = FunctionSpace (mesh, Qk)

        uinter = interpolate (uexact , W)
        er_interpL2 = np.sqrt(assemble((uexact - w)**2 *dx))
        Qerrors_interp[io].append(er_interpL2)
```

    
       ::> Begin computations for order 1 and refinement 40
        |--Number of vertices = 1681
        |--Total number of unknowns = 1681
    
      ::> Elapsed time:  0:00:00.157575 
    
    
       ::> Begin computations for order 1 and refinement 80
        |--Number of vertices = 6561
        |--Total number of unknowns = 6561
    
      ::> Elapsed time:  0:00:00.747820 
    
    
       ::> Begin computations for order 1 and refinement 160
        |--Number of vertices = 25921
        |--Total number of unknowns = 25921
    
      ::> Elapsed time:  0:00:02.050600 
    
    
       ::> Begin computations for order 2 and refinement 40
        |--Number of vertices = 1681
        |--Total number of unknowns = 6561
    
      ::> Elapsed time:  0:00:00.258165 
    
    
       ::> Begin computations for order 2 and refinement 80
        |--Number of vertices = 6561
        |--Total number of unknowns = 25921
    
      ::> Elapsed time:  0:00:01.121458 
    
    
       ::> Begin computations for order 2 and refinement 160
        |--Number of vertices = 25921
        |--Total number of unknowns = 103041
    
      ::> Elapsed time:  0:00:04.835536 
    
    
       ::> Begin computations for order 3 and refinement 40
        |--Number of vertices = 1681
        |--Total number of unknowns = 14641
    
      ::> Elapsed time:  0:00:00.527316 
    
    
       ::> Begin computations for order 3 and refinement 80
        |--Number of vertices = 6561
        |--Total number of unknowns = 58081
    
      ::> Elapsed time:  0:00:02.083121 
    
    
       ::> Begin computations for order 3 and refinement 160
        |--Number of vertices = 25921
        |--Total number of unknowns = 231361
    
      ::> Elapsed time:  0:00:12.631979 
    



```python
Qelapsed_time_df = pd.DataFrame(data = [[str(time) for time in el] for el in Qelapsed_time],
                            columns = mesh_refinements,
                            index = [1,2,3])

QerrorsH1_df  = pd.DataFrame(data = QerrorsH1,
                            columns = mesh_refinements,
                            index = [1,2,3])


QerrorsL2_df  = pd.DataFrame(data = QerrorsL2,
                            columns = mesh_refinements,
                            index = [1,2,3])


Qerrors_interpL2_df  = pd.DataFrame(data = Qerrors_interp,
                                columns = mesh_refinements,
                                index = [1,2,3])


Qunknowns_df  = pd.DataFrame(data = Qunknowns,
                             columns = mesh_refinements,
                             index = [1,2,3])


Qhmax_df  = pd.DataFrame(data = Qhmax,
                             columns = mesh_refinements,
                             index = [1,2,3])

QerrorsH1_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.201532</td>
      <td>0.100740</td>
      <td>5.036670e-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.004085</td>
      <td>0.001021</td>
      <td>2.553389e-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000054</td>
      <td>0.000007</td>
      <td>8.474846e-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid = sns.lineplot(data=QerrorsH1_df)
grid.set(xscale="linear", yscale="log")
```




    [None, None]




![png](output_25_1.png)



```python
QerrorsL2_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.815441e-03</td>
      <td>4.539573e-04</td>
      <td>1.134953e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.575503e-05</td>
      <td>1.969835e-06</td>
      <td>2.462438e-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.480865e-07</td>
      <td>9.266789e-09</td>
      <td>5.794884e-10</td>
    </tr>
  </tbody>
</table>
</div>




```python
Qerrors_interpL2_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.815441e-03</td>
      <td>4.539573e-04</td>
      <td>1.134953e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.575503e-05</td>
      <td>1.969835e-06</td>
      <td>2.462438e-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.480865e-07</td>
      <td>9.266789e-09</td>
      <td>5.794884e-10</td>
    </tr>
  </tbody>
</table>
</div>




```python
Qelapsed_time_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0:00:00.058449</td>
      <td>0:00:00.176032</td>
      <td>0:00:00.770131</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0:00:00.119010</td>
      <td>0:00:00.446479</td>
      <td>0:00:01.804615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0:00:00.237913</td>
      <td>0:00:00.954094</td>
      <td>0:00:04.679822</td>
    </tr>
  </tbody>
</table>
</div>




```python
elapsed_time_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0:00:00.179522</td>
      <td>0:00:00.786902</td>
      <td>0:00:02.312224</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0:00:00.400928</td>
      <td>0:00:02.155824</td>
      <td>0:00:07.796583</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0:00:01.549421</td>
      <td>0:00:05.015535</td>
      <td>0:00:19.248418</td>
    </tr>
  </tbody>
</table>
</div>




```python
unknowns_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3243</td>
      <td>12845</td>
      <td>50925</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12786</td>
      <td>50875</td>
      <td>202939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28630</td>
      <td>114091</td>
      <td>456043</td>
    </tr>
  </tbody>
</table>
</div>




```python
Qunknowns_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1681</td>
      <td>6561</td>
      <td>25921</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6561</td>
      <td>25921</td>
      <td>103041</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14641</td>
      <td>58081</td>
      <td>231361</td>
    </tr>
  </tbody>
</table>
</div>




```python
Qhmax_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.035355</td>
      <td>0.017678</td>
      <td>0.008839</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.035355</td>
      <td>0.017678</td>
      <td>0.008839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.035355</td>
      <td>0.017678</td>
      <td>0.008839</td>
    </tr>
  </tbody>
</table>
</div>



### Solve for the circle

Solve the problem in a circular region of radius R = 1, $\kappa = 1$, f = 1 and boundary conditions $u = 0$, for which the exact solution is

$$
\frac{1 - x_1^2 - x_2^2}{4}
$$


```python
radius = 1
domain = Circle(Point(0,0), radius)
mesh_refinement = 10
order = 2
uexact = Expression("(1 - pow(x[0],2) - pow(x[1],2))/4", degree = 2)


# For remembering relevant info
Celapsed_time  = []
Cerrors1H1     = []
Cerrors1L2     = []
Cerrors2H1     = []
Cerrors2L2     = []
Cerrors_interp = []
Cunknowns      = []
Chmax          = []

# Order of finite elements space
orders = [1,2,3]
mesh_refinements = [40,80,160]

for io, order in enumerate(orders):
    Celapsed_time.append([])
    Cerrors1H1.append([])
    Cerrors1L2.append([])
    Cerrors2H1.append([])
    Cerrors2L2.append([])
    Cerrors_interp.append([])
    Cunknowns.append([])
    Chmax.append([])
    
    for im, mesh_refinement in enumerate(mesh_refinements):

        Pk = FiniteElement("Lagrange", 'triangle', order)
        mesh = generate_mesh(domain, mesh_refinement)


        # Finite element space
        W = FunctionSpace(mesh, Pk)


        ### Variational formulation: Poisson problem
        u = TrialFunction(W)
        v = TestFunction(W)

        funcdom = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
        dx = Measure("dx")(subdomain_data=funcdom)


        # Volumetric source
        f = Constant(1)

        a = inner(kappa*grad(u), grad(v))*dx
        L = f*v*dx

        #### Dirichlet boundary conditions on W
        bcs = DirichletBC(W, Constant(0), DomainBoundary())

        ### Solution
        w1 = Function(W)
        solve(a == L, w1, bcs)

        #print("\n With zeroed boundary condition")
        #print("| - Error L2 = ", errornorm(uexact, w1, 'L2'))
        #print("| - Error H1 = ", errornorm(uexact, w1, 'H1'))
                
        Cerrors1H1[io].append(errornorm(uexact, w1, 'H1'))
        Cerrors1L2[io].append(errornorm(uexact, w1, 'L2'))        

        Cunknowns[io].append( W.dim())
        Chmax[io].append(mesh.hmax())
       
        
        # Fixed boundary conditions, using exact values
        bcs = DirichletBC(W, uexact, DomainBoundary())

        ### Solution
        w2 = Function(W)

        solve(a == L, w2, bcs)

        #print("\n With true boundary condition")
        #print("| - Error L2 = ", errornorm(uexact, w2, 'L2'))
        #print("| - Error H1 = ", errornorm(uexact, w2, 'H1'))
        
        Cerrors2H1[io].append(errornorm(uexact, w2, 'H1'))
        Cerrors2L2[io].append(errornorm(uexact, w2, 'L2'))
```


```python
plt.figure(figsize=(15,15))
plt.subplot(121)
#plot(mesh, linewidth=.2)
plot(w2)

plt.subplot(122)
#plot(mesh, linewidth=.2)
plot(w1)
```




    <matplotlib.tri.tricontour.TriContourSet at 0x7f71da0d9710>




![png](output_35_1.png)



```python
CerrorsL2_df  = pd.DataFrame(data = Cerrors1L2,
                            columns = mesh_refinements,
                            index = [1,2,3])


CerrorsH1_df  = pd.DataFrame(data = Cerrors1H1,
                            columns = mesh_refinements,
                            index = [1,2,3])

Cerrors2L2_df  = pd.DataFrame(data = Cerrors2L2,
                            columns = mesh_refinements,
                            index = [1,2,3])


Cerrors2H1_df  = pd.DataFrame(data = Cerrors2H1,
                            columns = mesh_refinements,
                            index = [1,2,3])

CerrorsL2_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.000248</td>
      <td>0.000062</td>
      <td>0.000015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000186</td>
      <td>0.000047</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000185</td>
      <td>0.000046</td>
      <td>0.000012</td>
    </tr>
  </tbody>
</table>
</div>




```python
CerrorsH1_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.010167</td>
      <td>0.005070</td>
      <td>0.002505</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.001829</td>
      <td>0.000620</td>
      <td>0.000228</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.001488</td>
      <td>0.000525</td>
      <td>0.000186</td>
    </tr>
  </tbody>
</table>
</div>




```python
Cerrors2H1_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>40</th>
      <th>80</th>
      <th>160</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.003393e-02</td>
      <td>5.003364e-03</td>
      <td>2.495532e-03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.215833e-13</td>
      <td>2.853964e-12</td>
      <td>1.136096e-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.921608e-12</td>
      <td>7.618397e-12</td>
      <td>3.036043e-11</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid = sns.lineplot(data=Cerrors2H1_df)
grid.set(xscale="linear", yscale="log")
```




    [None, None]




![png](output_39_1.png)



```python
#Celapsed_time_df = pd.DataFrame(data = [[str(time) for time in el] for el in Celapsed_time],
#                            columns = mesh_refinements,
#                            index = [1,2,3])

CerrorsH1_df  = pd.DataFrame(data = Cerrors1H1,
                            columns = mesh_refinements,
                            index = [1,2,3])


CerrorsL2_df  = pd.DataFrame(data = Cerrors1L2,
                            columns = mesh_refinements,
                            index = [1,2,3])


Cerrors_interpL2_df  = pd.DataFrame(data = Cerrors_interp,
                                columns = mesh_refinements,
                                index = [1,2,3])


Cunknowns_df  = pd.DataFrame(data = Cunknowns,
                             columns = mesh_refinements,
                             index = [1,2,3])


Chmax_df  = pd.DataFrame(data = Chmax,
                             columns = mesh_refinements,
                             index = [1,2,3])

CerrorsH1_df
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _list_to_arrays(data, columns, coerce_float, dtype)
        496         result = _convert_object_array(
    --> 497             content, columns, dtype=dtype, coerce_float=coerce_float
        498         )


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _convert_object_array(content, columns, coerce_float, dtype)
        580             raise AssertionError(
    --> 581                 f"{len(columns)} columns passed, passed data had "
        582                 f"{len(content)} columns"


    AssertionError: 3 columns passed, passed data had 0 columns

    
    The above exception was the direct cause of the following exception:


    ValueError                                Traceback (most recent call last)

    <ipython-input-198-77c1fe7c181c> in <module>
         15 Cerrors_interpL2_df  = pd.DataFrame(data = Cerrors_interp,
         16                                 columns = mesh_refinements,
    ---> 17                                 index = [1,2,3])
         18 
         19 


    ~/.local/lib/python3.6/site-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
        472                     if is_named_tuple(data[0]) and columns is None:
        473                         columns = data[0]._fields
    --> 474                     arrays, columns = to_arrays(data, columns, dtype=dtype)
        475                     columns = ensure_index(columns)
        476 


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in to_arrays(data, columns, coerce_float, dtype)
        459         return [], []  # columns if columns is not None else []
        460     if isinstance(data[0], (list, tuple)):
    --> 461         return _list_to_arrays(data, columns, coerce_float=coerce_float, dtype=dtype)
        462     elif isinstance(data[0], abc.Mapping):
        463         return _list_of_dict_to_arrays(


    ~/.local/lib/python3.6/site-packages/pandas/core/internals/construction.py in _list_to_arrays(data, columns, coerce_float, dtype)
        498         )
        499     except AssertionError as e:
    --> 500         raise ValueError(e) from e
        501     return result
        502 


    ValueError: 3 columns passed, passed data had 0 columns


### Bônus


```python

alpha = 1

# Define function spaces and mixed (product) space
V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
W = V * R

# Define variational problem
(u, l) = TrialFunction(W)
(v, d) = TestFunctions(W)
f = Expression("8*pow(pi,2)*cos(2*pi*x[0])*cos(2*pi*x[1])", degree = 5)
a = (inner(grad(u), grad(v)) + l*v)*dx
L = f*v*dx
```
