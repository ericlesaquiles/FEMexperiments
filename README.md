# Some playing with finite element methods in Fenics

There follows in the folders ex1, ex2, ex3 and ex4 four examples,
written in `Jupyter notebook`, in
somewhat increasing order of complexity (but not quite) four examples
of applications of finite element methods solver library (fenics).

More specifically, we have

1. In ex1, it's tackled the Poisson problem, but with non-homogeneous
thermal diffusivity constants (or rather, they're constant in the four
defined subregions - check the graphs to see what I mean). It's a
simple enough example to get you started, but also shows how easy it
is to deal with a "inhomogeneous object";

2. In ex2, we come back to the Poisson problem, but with different
boundary conditions. In this notebook, we make go into greater detail
into measuring the time and the error for different mesh sizes and
different orders of interpolating polynomials

3. In ex3 we tackle the Diffusion-Convection equation, having a case
when refining the mesh or uniformly increasing the orders of polynomials might
not help much, and different "tricks" are needed;

4. In ex4 we tackle the elasticity problem, with different geometries,
and showing some "tricks" to better deal with symmetric geometries.

They were intended for mostly playing purposes, so I apologize if the
notebooks are lacking in some points (and they are).

I intend to come back and work a little more, to make them perhaps
more useful - but until then, if you have any question, just let me know.