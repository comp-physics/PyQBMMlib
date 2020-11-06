# PyQBMMlib

PyQBMMlib is a Python fork of [QBMMlib](https://github.com/sbryngelson/QBMMlib), which was developed by Dr. Spencer Bryngelson, Prof. Rodney Fox, and Prof. Tim Colonius. 
It can be cited as
```
@article{bryngelson_2020,
    Author        = {Spencer H. Bryngelson and Tim Colonius and Rodney O. Fox},
    Title         = {{QBMMlib: A} library of quadrature-based moment methods},
    Volume        = 12
    Pages         = 100615
    Year          = {2020},
}
```
Its documentation is also located on the [arXiv](http://arxiv.org/abs/2008.05063v1).
When compared to QBMMlib, PyQBMMlib offers significantly decreased time to solution (when using Numba).

## Authors: 

* [Spencer H. Bryngelson](https://bryngelson-research.com) (Caltech) 
    * spencer@caltech.edu
* Esteban Cisneros (Illinois)
    * csnrsgr2@illinois.edu

## Required Python modules

- Python >= 3.0
- Numpy
- Scipy
- Sympy
- Optional: Numba (significant speed increase via JIT compiling)

## Current capabilities 

PyQBMMlib is under active development.
However, it still has all the capabilities of QBMMlib except for traditional CQMOM, which was elided in lieu of its contemporary, CHyQMOM.
This includes:
- Automatic formulation of moment transport equations
- 1-3D moment inversion
- QMOM (Wheeler), HyQMOM (2 and 3 node), CHyQMOM (4, 9,  and 27 node)
- SSP RK2-3 

## Features under development

Several more features will be added to PyQBMMlib.
A partial list is included here.
- 2D + static 1D moment inversion
- Spatial dependencies and fluxes (3D flows)

## Acknowledgements

Great thanks is owed to Professor Alberto Passalacqua (Iowa State University) for his part in developing these algorithms and teaching me the same.
We acknowledge funding from the U.S. Office of Naval Research under grant numbers N0014-17-1-2676 and N0014-18-1-2625 (SHB) and the U.S. Department of Energy, National Nuclear Security Administration, under Award Number DE-NA0002374 (ECG).

