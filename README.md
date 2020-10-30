# PyQBMMlib

Developers: Spencer H. Bryngelson (Caltech), Esteban Cisneros (Illinois)

This is a Python version of [QBMMlib](https://github.com/sbryngelson/QBMMlib), which was developed by Dr. Spencer Bryngelson, Prof. Rodney Fox, and Prof. Tim Colonius. 
It can be cited as
```
@article{bryngelson_2020,
    Author        = {Spencer H. Bryngelson and Tim Colonius and Rodney O. Fox},
    Title         = {QBMMlib: A library of quadrature-based moment methods},
    Journal       = {SoftwareX},
    Year          = {2020},
}
```
Its documentation is also located on the [arXiv](http://arxiv.org/abs/2008.05063v1).

## Requirements

- Python >= 3.0
- Numpy
- Scipy
- Sympy
- Optional: Numba (significant speed increase via JIT compiling)

## Current capabilities 

PyQBMMlib currently has all the capabilities of QBMMlib except for traditional CQMOM, which was elided in lieu of CHyQMOM.
This includes:
- Automatic formulation of moment transport equations
- 1-3D moment inversion
- QMOM (Wheeler), HyQMOM (2 and 3 node), CHyQMOM (4, 9,  and 27 node)
- SSP RK2-3 

## Features under development

- 2D + static 1D moment inversion
- Spatial dependencies and fluxes (3D flows)
