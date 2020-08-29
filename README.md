# PyQBMMlib

Spencer Bryngelson, Esteban Cisneros

## Functionality

1. Get carried moment indices/exponents for a given number of quadrature points and moment inversion method
    (`MomentIndex` in QBMMlib Mathematica)
    - For 1D problems the returned array in Mathematica QBMMlib will be 1D (a list of required indices, but since there is only 1 internal coordinate there can only be one exponent per 'index')
    - For 2D problems the returned array will be a list that is $N$ moments long, with each moment having 2 components (one for each internal coordinate).
        - See below [number 2] for how these components show up
2. Differential equation + pop. balance equation = Moment transport equations
    (`TransportTerms` in QBMMlib)
    In QBMMlib, `TransportTerms` exports the required moment indices for the RHS of the moment trasport equations `[exps]` and the coefficients that multiply each of these tearms `[coeffs]`
    - Exponents refer to the moment indices themselves.
        - For example: $M_{1,2} = \int f(x,y) x^1 y^2 dx dy$, where $M_{1,2}$ is the one-two moment
        - So: $M_{i,j} = \int f(x,y) x^i y^j dx dy$
            - Note that $f(x,y)$ is called the NDF number density function. It is basically a scaled PDF.
        - Here, $(i,j)$ are the "exponents" `[exps]`, you might also call them moment indices, and $(x,y)$ are the internal coordinates
        - The carried moment set always uses integer sets of $(i,j)$, however the moments computed via quadrature at the end (step 4) can have non-integer $i$ and $j$.
        - In practice, each of these moments have coefficients out front of them `[coeffs]`
3. Invert a moment set into a set of quadrature points and weights using an inversion algorithm [e.g. Wheeler, CQMOM, etc.]
    (`Moment Invert` in QBMMlib)
4. Use quadrature to approximate certain moments 
    (`ComputeRHS` and `Project` in QBMMlib)
5. Time step
    (Adaptive time marching using SSP RK2/3 [embedded])

## Architecture

- Items 1, 3, 4, and 5 can be more-or-less copied directly to Python code.

- Item 2 will require SymPy or something like it.
However, example moment equations can be generated via QBMMlib and used for testing.

- Items 1 and 3 are coupled.
The choice of inversion algorithm decides what moments need to be carried and how the inversion is done.
Thus, this parameter choice needs to be made up front.

- Their are several different inversion algorithms.
They can be ported directly from QBMMlib.
They have restrictions: some are only for 1D problems (Wheeler, adaptive wheeler, HYQMOM) and others are for 2+D problems (CQMOM, CHYQMOM).
There are other algorithms that can also be implemented, but these are the ones already in QBMMlib.
Some of the 2+D algorithms will call the 1D algorithms themselves, so they need to be able to communicate.

- Time steppers are agnostic to all of these things and can be entirely separate

## First steps

I think it is best to start with the easiest things and work up from there.

### Step 1: Moment inversion

0. Determine what moments ($i$) are required for your moment inversion algorithm 
1. Generate example moment set via a 1D distribution (e.g. Gaussian)
2. Invert this moment set to a set of quadrature weights and nodes using the 1D methods (Wheeler, etc.)
3. Confirm they agree with Mathematica code
4. Repeat this process with a 2D distribution 
    - Determine required moment pairs $(i,j)$
    - Initializing moment set using, e.g., bivariate Gaussian
    - Invert to quadrature weights/nodes using CQMOM and CHyQMOM
    - Let's use $(l,m)$ as indices for the weight/node pairs

### Step 2: Quadrature

$$ M_{i,j} = \sum_{l,m} w_{l,m} x_1(l)^i x_{2}(m)^j $$
where $l,m$ are indices of the quadrature weights $\bm{w}$ and abscissas $\bm{x} = \{ x_1, x_2 \}$

- Use above quadrature to show that you can 
    1. Reproduce in-moment-set moments down to round-off
    2. See that you get some errors when trying to approximate higher order moments (not-in-moment-set)

### Step 3: Get the moment transport equation

1. Get an example set of moment transport equations from QBMMlib
    - This consists of moment exponents and coefficients, that, when summed, give the right-hand-side of each moment transport equation.
2. Use a loop to compute, multiply, then sum these terms together
3. Return the result as the RHS

### Step 4: Time integrate

1. Obvious...

