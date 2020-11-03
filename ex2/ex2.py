#!/usr/bin/env python
# coding: utf-8

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

# ### User defined parameters:
# #### Set the following parameters according to your problem

# avoid having _useless_ printint on screen


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

Qk = False
elapsed_times = []
elapsed_time2 = []
elapsed_time3 = []
times = 3


h1error = []

orders = [1,2,3]
mesh_refinements = [40,80]

run_rectangle = True
run_circle    = False

if run_rectangle:
    for t in range(times):
        elapsed_times.append([])
        for io, order in enumerate(orders):
            elapsed_times[t].append([])
            h1error.append([])

            for im, mesh_refinement in enumerate(mesh_refinements):
                if Qk:
                    Pk = FiniteElement("Lagrange", 'quadrilateral', order)
                else:
                    Pk = FiniteElement("Lagrange", 'triangle', order)

                #### IO setup
                ufile_pvd  = File(dir_+f'/temperatureO{order}mr{mesh_refinement}.pvd')
                domfile_pvd = File(dir_+f"/auxfuncO{order}mr{mesh_refinement}.pvd")
            
                startTime = datetime.now()
                print('\n   ::> Begin computations')


                #### Mesh generation
            
                if Qk:
                    mesh = UnitSquareMesh.create(mesh_refinement, mesh_refinement, CellType.Type.quadrilateral)
                else:
                    mesh = generate_mesh(domain, mesh_refinement)
                print("    |-Mesh done")
                print("    |--Number of vertices = "+str(mesh.num_vertices()))
                print("    |--Number of cells = "+str(mesh.num_cells()))
                print("    |--Cell size hmax,hmin = %.3g %.3g" % (mesh.hmax(), mesh.hmin()))


                # Finite element space
                W = FunctionSpace(mesh, Pk)
                print("    |--Total number of unknowns = %d" % (W.dim()))

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

                # Fixed boundary temperature: left, right, top, bottom
                Tboundary = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])", degree = 5)
                bcs = DirichletBC(W, Tboundary, DomainBoundary())


                ### Solution
                w = Function(W)

                # For Mesh refinement
                problem = LinearVariationalProblem(a, L, w, bcs)
                solver = LinearVariationalSolver(problem)
                prm = solver.parameters
                linear_solver = 'Krylov'
                if linear_solver == 'Krylov':
                    prm ["linear_solver"]= "gmres"
                    prm ["preconditioner"]= "ilu"
                    prm ["krylov_solver"]["absolute_tolerance"]= 1e-11
                    prm ["krylov_solver"]["relative_tolerance"]= 1e-10
                    prm ["krylov_solver"]["maximum_iterations"]= 10000
                    prm ["krylov_solver"]["monitor_convergence"]= True
                    prm ["krylov_solver"]["nonzero_initial_guess"]= True
                else :
                    prm ["linear_solver"]= "mumps"

                # solver.solve()
                # solve(a == L, w, bcs,
                #       solver_parameters={'linear_solver':'cg', 'preconditioner':'ilu'})
                solve(a == L, w, bcs)
                
                ####### IO
                ufile_pvd << w
                domfile_pvd << funcdom

                elapsed_times[t][io].append(datetime.now() - startTime)
                #print('\n  ::> Elapsed time: ', str(elapsed_times[t][io][im]), '\n')


                # plt.figure(figsize=(15,15))
                # plot(mesh,linewidth=.2)
                # c = plot(w, wireframe=True, title=f'Solution')
                # plt.colorbar(c)
                # plt.show()

                #### Compute errors
                uexact = Expression('cos(2*pi*x[0])*cos(2*pi*x[1])', degree = 7)
                h1error[io].append(errornorm(uexact, w, 'H1'))
      
                print("| - Error L2 = ", errornorm(uexact, w, 'L2'))
                print("| - Error H1 = ", errornorm(uexact, w, 'H1'))

                # Showing results 
                # plt.figure()       # create a plot figure
                # plt.subplot(3,3,1) # (rows, columns, panel number)


                # Interpolate exact solution and compute interpolation error
                Pk = FiniteElement("Lagrange", 'triangle', order)
                W = FunctionSpace (mesh, Pk)

                uinter = interpolate ( uexact , W )
                print("| - Interpolate Error L2 = ", errornorm(uexact, w, 'L2'))
                print("| - Interpolate Error H1 = ", errornorm(uexact, w, 'H1'))

# elapsed_time = pd.DataFrame(data = [[time.microseconds for time in el] for el in elapsed_time],
#                             columns = mesh_refinements,
#                             index = [1,2,3])

# errorsh1_df  = pd.DataFrame(data = h1error,
#                             columns = mesh_refinements,
#                             index = [1,2,3])




















###################################################################
############### Solve for the circle

if run_circle:

    radius = 1
    domain = Circle(Point(0,0), radius)
    mesh_refinement = 10
    order = 2
    uexact = Expression("(1 - pow(x[0],2) - pow(x[1],2))/4", degree = 2)


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

    print("| - Error L2 = ", errornorm(uexact, w1, 'L2'))
    print("| - Error H1 = ", errornorm(uexact, w1, 'H1'))



    # Fixed boundary conditions, using exact values
    bcs = DirichletBC(W, uexact, DomainBoundary())

    ### Solution
    w2 = Function(W)

    solve(a == L, w2, bcs)

    print("| - Error L2 = ", errornorm(uexact, w2, 'L2'))
    print("| - Error H1 = ", errornorm(uexact, w2, 'H1'))






#################################################
########## Solving the projection problem


alpha = 1

# Define function spaces and mixed (product) space
# V = FunctionSpace(mesh, "CG", 1)
# R = FunctionSpace(mesh, "R", 0)
# W = V * R

# Define variational problem
# (u, l) = TrialFunction(W)
# (v, d) = TestFunctions(W)
# f = Expression("8*pow(pi,2)*cos(2*pi*x[0])*cos(2*pi*x[1])", degree = 5)
# a = (inner(grad(u), grad(v)) + l*v)*dx
# L = f*v*dx
