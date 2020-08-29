(* ::Package:: *)

(* 

This file is part of QBMMlib

QBMMlib is the legal property of its developers, whose names 
are listed in the copyright file included with this source 
distribution.

QBMMlib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published 
by the Free Software Foundation, either version 3 of the license 
or any later version.

QBMMlib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with QBMMlib (LICENSE).  
If not, see <http://www.gnu.org/licenses/>.

*) 

BeginPackage["QBMMlib`"];

TransportTerms::usage="
    TransportTerms[eqn,depvar,indvar] computes the coefficients and exponents of the internal coordinates that correspond to the RHS operator of the moment transport equations.
    Syntax follows NDSolve and the like.
    'eqn' is the governing ODE, 'depvar' is the dependent variable (e.g. x[t]), and 'indvar' is the independent variable (e.g. t).
    One free variable is allowed, which can be conditioned upon for a 3D moment method.
";

MomentIndex::usage = "
    MomentIndex[n,method,nperm] computes the moments required by the 'method' (QMOM, AQMOM, CQMOM, CHYQMOM) for a given number of nodes n. 
    n is expected to be a list with length matching the length of the number of internal coordinates.
    Extra moments are included if the number of permuations, 'numperm', requires it.
    Constraints: In 2D CHyQMOM only supports n[[1]] == n[[2]]. 
";

Wheeler::usage="
    Wheeler[mom] computes the one-dimensional optimal weights and abscissa given moments (mom) using Wheeler's algorithm [Wheeler, Rocky Mt. J. Math. 1974].
";

MomentInvert::usage = "
    MomentInvert[moms,ks,Method->method,Permutation->perm,MaxSkewness->maxskew] computes the optimal weights (w) and abscissas (xi) given a moment set (moments) with corresponding indices (ks).
    The moment inversion algorithm (method) is used.
    (method) is one of CQMOM [Yuan & Fox JCP 2011], CHyQMOM [Fox et al., JCP 2018; Patel et al., JCPX 2019], HyQMOM (1D only) or QMOM (1D only) [Wheeler, Rocky Mt. J. Math. 1974].
    (perm) is the permutation [=12 for v1 | v2 or = 21 for v2 | v1] for two-dimensional problems.
    For method='CHYQMOM' and n[[1]]=n[[2]]=3 or method='HYQMOM' and n[[1]]=3 a maximum skewness can be prescribed via option 'MaxSkewness->maxskew'
    By default (maxskew) = 30.
";

Quadrature::usage="
    Quadrature[w,xi,idx] computes the raw moment M = Int[ Product[ w_i (x)_i^(idx)_i d(x)_i ]  P ] (for indices idx, abscissas xi, and weights w). 
";

Project::usage="
    Project[w,xi,idxs] computes all moments in (idxs) via quadrature (Quadrature) using weights (w) and abscissas (xi).
";

MomentFind::usage="
    MomentFind[mom,idxs,idx] returns the (idx) = {idx_1,...,idx_N} moment, where (N) is the number of internal coordinates, from list of moments (mom) and indices (idxs), so long as idx is a component of indxs.
";

ComputeRHS::usage="
    ComputeRHS[w,xi,idxs,{coef,exp},Conditioning->xi3] computes the RHS of the moment transport equations.
    (w) and (xi) are the weights and abscissas, (idxs) are the indices of the moment set, (coef) and (exp) are the coefficients and exponents associated with the internal coordinates required to close the equations
    (coef) and (exp) should be computed using TransportTerms.
    ComputeRHS operates natively for 1D and 2D moment sets, and can take an optional third coordinate direction for which the abscissa are fixed (xi3) with option Conditioning->xi3.
"

RK23::usage="
    RK23[mom,F,t,dt] uses the strong-stability preserving (SSP), third-order accurate, Runge--Kutta algorithm to update the moments (mom) according to right-hand-side operator (F) at time (t) with time step (dt). 
    The L1 norm of the difference between this and the solution using an embedded SSP second-order accurate Runge--Kutta algorithm is used to compute the time step error to first-order accuracy.
    The output is {momnew,e}, where (momnew) are the updated moments and (e) is the approximation of time-step error.
";

OutAbscissa::usage="
    OutAbscissa[xi] returns a threaded list of the abscissa (xi) in 1D, 2D, or 3D (as appropritate).
"

Begin["`Private`"];

(* pow::usage="
    pow[x,y] returns x^y where x^0 = 1, even if x=0
"; *)

pow[x_, 0] := 1 /; x == 0 || x == 0.
pow[x_, y_: 0] := x^y

MomentFind[moms_, ks_, idx_] := Module[
    {loc}, 
    loc=Position[ks,idx][[1,1]];
    If[IntegerQ[loc],Return[moms[[loc]],Module];];
    Print["Error: You are looking for a moment that does not exist in your moment set."];Abort[]; 
];



TransportTerms[eqn_, rt_, dep_] := Module[
   {v, integrand, dim, mrdd, list, exp, coefs, exps, vars, l, m, n, 
    freevar, invars, idx, nterms},
   
   (* How many derivatives do we have? *)
   Do[
        If[MemberQ[eqn,D[rt,{dep,i}], {0,-1}, Heads->True], dim = i];
    ,{i,10}];
    If[dim > 2,Print["Error: Highest order derivate > 2. Its order: ", dim];Abort[];];
   
    (* Free variables, get rid of the dependent variable *)
    freevar = DeleteCases[DeleteDuplicates@Cases[eqn, _Symbol, Infinity], dep];
    freevar = DeleteCases[freevar, _?NumericQ, Infinity];
    If[Length[freevar] == 1, dim = 3];
    If[Length[freevar] > 1, 
        Print["Error: Too many free variables in governing equation: ", freevar]; Abort[];
    ];
   
    idx = Table[Symbol["c"][i], {i, dim}];
    If[dim == 1, 
        vars = {v[1]};
        {l} = idx;
        v[1] = rt;
        v4 = D[rt, dep];
        invars = {v[1]};
    ];
    If[dim == 2, 
        vars = {v[1], v[2]};
        {l, m} = idx;
        v[1] = rt;
        v[2] = D[rt, dep];
        v4 = D[rt, {dep, 2}];
        invars = {v[1], v[2]};
     ];
     If[dim == 3, 
        vars = {v[1], v[2], v[3]};
        {l, m, n} = idx;
        v[1] = rt;
        v[2] = D[rt, dep];
        v[3] = freevar[[1]];
        v4 = D[rt, {dep, 2}];
        invars = {v[1], v[2], v[3]};
     ];
   
    If[dim != 1 && dim != 2 && dim != 3, 
        Print["Error: Dimesionallity not supported. You used dim = ", dim]; Abort[];
    ];
   
    mrdd=v4/.Solve[eqn,v4][[1]];

    If[dim==1,integrand=mrdd v[1]^(l-1)];
    If[dim==2,integrand=mrdd v[1]^l v[2]^(m-1)];
    If[dim==3,integrand=mrdd v[1]^l v[2]^(m-1) v[3]^n];

    nterms = Length[Expand[integrand] + Unique["x"]] - 1;
    If[nterms==1,
        list={integrand},
        list=PowerExpand[List@@Distribute[integrand]]
    ];
    exps=Table[Exponent[list[[i]],j],{i,Length[list]},{j,invars}];

    If[dim==1,
        coefs=l Table[Coefficient[list[[i]],Product[v[j]^exps[[i,j]],{j,dim}]],{i,Length[list]}];
    ,
        coefs=m Table[Coefficient[list[[i]],Product[v[j]^exps[[i,j]],{j,dim}]],{i,Length[list]}];
    ];

    (* Add term that isn't in the integrand *)
    If[dim == 2, AppendTo[exps, {l - 1, m + 1}]];
    If[dim == 3, AppendTo[exps, {l - 1, m + 1, n}]];
    If[dim > 1, AppendTo[coefs, l]];
    Return[{coefs, exps}, Module];
];


OutAbscissa[xi_] := Module[{nro,dim},
    If[NumberQ[Max[xi[[1]][1]]],
        dim=3;,
        dim=Length[Dimensions[xi]];
    ];

    If[dim!=1&&dim!=2&&dim!=3,
        Print["Dimesionallity incorrect! Dim = ",dim];Abort[];
    ];

    If[dim == 1, Return[Re[Flatten[xi]],Module]];
    If[dim == 2, Return[Re[Thread[xi]]]];
	If[dim == 3, 
        nro = 0;
        Do[
            If[
                NumberQ[Max[xi[[1]][k]]],
                nro=k,
                Break[]
            ],
        {k,10^5}];
        Return[Table[Thread[{xi[[1]][j], xi[[2]][j]}], {j,nro}],Module]
    ];
];

Options[MomentInvert] = {Method->"CHYQMOM",Permutation->12,MaxSkewness->30};
MomentInvert[ mom_, ks_, opts : OptionsPattern[]] := Module[{method,perm,xi,w,dim,maxskew},
    dim = Dimensions[ks][[2]];
    method = OptionValue[Method];
    perm   = OptionValue[Permutation];
    maxskew= OptionValue[MaxSkewness];
    If[dim==1,
        If[method=="HYQMOM",Return[HYQMOM[mom,maxskew],Module]];
        If[method=="QMOM",
            {xi,w}=Wheeler[mom];
            Return[{w,xi},Module]
        ];
        If[method=="AQMOM",
            {xi,w}=AdaptiveWheeler[mom];
            Return[{w,xi},Module]
        ];
    ];
    If[dim==2||dim==3,
        If[method=="CHYQMOM",Return[CHYQMOM[mom,ks,maxskew],Module]];
        If[method=="CQMOM",Return[CQMOM[mom,ks,perm],Module]];
    ];
    Print["Error: method/dimensionality mismatch!"];Abort[];
];


Project[ w_, xi_, ks_] := Module[{mom,q,p,l,dim},
    dim=Length[ks[[1]]];
    If[dim==1,proj=Table[Quadrature[w, xi, {ks[[i]]}], {i, Length[ks]}]];
    If[dim==2,proj=Table[Quadrature[w, xi, {ks[[i, 1]], ks[[i, 2]]}], {i, Length[ks]}]];
    If[dim==3,proj=Table[Quadrature[w, xi, {ks[[i, 1]], ks[[i, 2]], ks[[i, 3]] + 1}], {i, Length[ks]}]];
    Return[proj,Module];
];

Quadrature[ w_, xi_, exp_] := Module[{nro,mom,q,p,l,dim},
    dim = Length[exp];
    If[dim==1,
        nro=0;
        p=exp[[1]];
    ];
    If[dim==2,
        nro=0;
        p=exp[[1]];
        q=exp[[2]];
    ];
    If[dim==3,
        nro=1;
        p=exp[[1]];
        q=exp[[2]];
        l=exp[[3]];
    ];
    If[dim==1,mom=Sum[w[[i]] xi[[i]]^p ,{i,Length[w]}]];
    If[dim==2,mom=Sum[w[[i]] xi[[1]][[i]]^p xi[[2]][[i]]^q,{i,Length[w]}]];
    If[dim==3,mom=Sum[w[l][[i]] xi[[1]][l][[i]]^p xi[[2]][l][[i]]^q,{i,Length[w[l]]}]];
	Return[mom,Module];
];

Options[ComputeRHS] = {Conditioning -> 0.0};
ComputeRHS[w_, xi_, ks_, terms_, opts : OptionsPattern[]] := Module[
   	{rhs, rule, myexps, mycoefs, rhstot, Ros, dim, exps, coefs},
   	Ros = OptionValue[Conditioning];
   	dim = Length[ks[[1]]];
	coefs = terms[[1]];
	exps = terms[[2]];
   	rhstot = {};
   	Do[
        rule = Table[Symbol["c"][i] -> ks[[j, i]], {i, dim}]; 
    	myexps = Table[exps[[i]] /. rule, {i, Length[exps]}];
    	mycoefs = Table[coefs[[i]] /. rule, {i, Length[coefs]}];
    	If[dim == 1 || dim == 2,
     	    rhs = Sum[
                mycoefs[[i]] Quadrature[w, xi, myexps[[i]]]
            ,{i,Length[coefs]}]
     	];
    	If[dim == 3,
            rhs = Sum[
        	    mycoefs[[i]] pow[Ros[[ks[[j, 3]] + 1]], myexps[[i, 3]] - ks[[j, 3]]]
                Quadrature[w, xi, {myexps[[i, 1]], myexps[[i, 2]], 1 + ks[[j, 3]]}]
        	,{i,Length[coefs]}];
     	];
    	AppendTo[rhstot, rhs];
    ,{j,Length[ks]}];
   	Return[rhstot, Module];
];

Wheeler[mom_] := Module[{mm = mom, nn, sig, a, b, Ja, w, xi, eval, evec, esys}, 
    mm = Flatten[mm];
    nn=Length[mm]/2;
    sig = Table[0., {i, 2 nn}, {j, 2 nn}]; 

    Do[sig[[2, i + 1]] = mm[[i + 1]], {i, 0, 2 nn - 1}]; 
    a = Table[0., {i, nn}]; b = a; a[[1]] = mm[[2]]/mm[[1]]; 
    b[[1]] = 0; 
    Do[
        Do[
            sig[[i + 2, j + 1]] = sig[[i + 1, j + 2]] - a[[i]] sig[[i + 1, j + 1]] - b[[i]] sig[[i, j + 1]];
            a[[i + 1]] = -(sig[[i + 1, i + 1]]/sig[[i + 1, i]]) + sig[[i + 2, i + 2]]/sig[[i + 2, i + 1]]; 
            b[[i + 1]] = sig[[i + 2, i + 1]]/sig[[i + 1, i]];
        ,{j,i, 2 nn - i - 1}];
    ,{i,nn - 1}];
    Ja = DiagonalMatrix[a]; 
    Do[
        Ja[[i, i + 1]] = -Sqrt[Abs[b[[i + 1]]]]; 
        Ja[[i + 1, i]] = -Sqrt[Abs[b[[i + 1]]]];
    ,{i,nn-1}]; 
    esys = Eigensystem[Ja]; 
    eval = esys[[1]]; 
    evec = esys[[2]]; 
    w = Table[evec[[i,1]]^2 mm[[1]],{i,nn}]; 
    Return[{eval, w}, Module];
];

MomentIndex[n_, method_, numperm_: 1] := Module[{ks, k1, k2, kstemp, nr, nrd, nro, dim}, 
   
    dim=Length[n];
    If[dim!=1&&dim!=2&&dim!=3,
        Print["Error: unsupported choice of dimensionality n ", n];Abort[];
    ];

    If[dim==1,
        nr  = n[[1]];
        If[method!="AQMOM"&&method!="QMOM"&&method!="HYQMOM",
            Print["Error: incompatible choices of method and dimensionality"];Abort[];
        ];
    ];
    If[dim==2,
        nr  = n[[1]];
        nrd = n[[2]];
        nro = 0;
        If[method!="CQMOM"&&method!="CHYQMOM",
            Print["Error: incompatible choices of method and dimensionality"];Abort[];
        ];
    ];
    If[dim==3,
        nr  = n[[1]];
        nrd = n[[2]];
        nro = n[[3]];
        If[method!="CQMOM"&&method!="CHYQMOM",
            Print["Error: incompatible choices of method and dimensionality"];Abort[];
        ];
    ];
    If[method == "QMOM" || method == "AQMOM",
        ks = Table[{i},{i,0,2 nr - 1}]; 
    ];
    If[method == "HYQMOM",
        If[nr==2,ks = {{0},{1},{2}}];
        If[nr==3,ks = {{0},{1},{2},{3},{4}}];
    ];
    If[method == "CHYQMOM", 
        If[nr == 2, 
            If[
                nro == 0, 
                ks = {{0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {0, 2}};
            ,
                ks = Flatten[Table[{{0, 0, i}, {1, 0, i}, {0, 1, i}, {2, 0, i}, {1, 1, i}, {0, 2, i}}, {i, 0, nro-1}], 1];
            ];
        ]; 
        If[nr == 3, 
            If[
                nro == 0, 
                ks = {{0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {0, 2}, {3, 0}, {0,3}, {4, 0}, {0, 4}}
            ,
                ks = Flatten[Table[{{0, 0, i}, {1, 0, i}, {0, 1, i}, {2, 0, i}, {1, 1, i}, {0, 2, i}, {3, 0, i}, {0,3, i}, {4, 0, i}, {0, 4, i}}, {i,0,nro-1}],1];
            ];
        ]; 
    ]; 
    If[method == "CQMOM", 
        If[nro == 0, 
            k1 = Flatten[{Table[{q,p}, {q, 0, nr - 1}, {p, 0, 2 nrd - 1}], Table[{q, p}, {q, nr, 2 nr - 1}, {p, 0, 0}]}, 2]; 
            k2 = Flatten[{Table[{p,q}, {q, 0, nrd - 1}, {p, 0, 2 nr - 1}], Table[{p, q}, {q, nrd, 2 nrd - 1}, {p, 0, 0}]}, 2]; 
            If[
                numperm==2,
                ks = DeleteDuplicates[Join[k1, k2]]; 
            ,
                ks = DeleteDuplicates[k1]; 
            ];
        ,
            Do[
                k1[i] = Flatten[{Table[{q,p,i}, {q, 0, nr - 1}, {p, 0, 2 nrd - 1}], Table[{q,p,i}, {q, nr, 2 nr - 1}, {p, 0, 0}]}, 2]; 
                k2[i] = Flatten[{Table[{p,q,i}, {q, 0, nrd - 1}, {p, 0, 2 nr - 1}], Table[{p,q,i}, {q, nrd, 2 nrd - 1}, {p, 0, 0}]}, 2]; 
                If[
                    numperm==2,
                    kstemp[i] = DeleteDuplicates[Join[k1[i],k2[i]]]; 
                ,
                    kstemp[i] = DeleteDuplicates[k1[i]]; 
                ];
            ,{i,0,nro-1}];
            ks=Flatten[Join[Table[kstemp[i],{i,0,nro-1}]],1];
        ];
    ]; 
    Return[ks, Module];
];
      
Pointer[ moms_, ks_ ] := Module[{eqns, vars, linsolv, pm, mymom}, 
    If[
        Length[ks[[1]]] == 2, 
        eqns = Table[moms[[i]] == pm[ks[[i, 1]], ks[[i, 2]]], {i,Length[ks]}]; 
        vars = DeleteDuplicates[ Flatten[Table[pm[ks[[i, 1]], ks[[i, 2]]], {i,Length[ks]}]]];
        linsolv = First[vars/.Solve[eqns, vars]];
        mymom[q_, p_] := linsolv[[First[First[Position[ks,{q,p}]]]]]; 
        Return[mymom, Module];
    ];
    If[
        Length[ks[[1]]] == 3,
        (* just tells you where to find each moments {qp,k}}, for each R_o,k  *)
        eqns = Table[moms[[i]] == pm[ks[[i, 1]], ks[[i, 2]], ks[[i, 3]]], {i,Length[ks]}]; 
        vars = DeleteDuplicates[ Flatten[Table[pm[ks[[i, 1]], ks[[i, 2]], ks[[i, 3]]], {i,Length[ks]}]]];
        linsolv = First[vars/.Solve[eqns, vars]];
        mymom[q_, p_, r_] := linsolv[[First[First[Position[ks,{q,p,r}]]]]]; 
        Return[mymom, Module];
    ]; 
    Print["Error: Index not 2D or 3D!"];Abort[];
];
 
(* Options[CHYQMOM] = {MaxSkewness->30}; *)
CHYQMOM[momin_, k_, q_:30] := Module[{moms = momin, ks = k},
    If[
        Length[k[[1]]]==2,
        If[Mod[Length[moms], 6] == 0,Return[CHYQMOM4m[moms, ks], Module]]; 
        If[Mod[Length[moms],10] == 0,Return[CHYQMOM9m[moms, ks, q], Module]];
    ,
        If[Mod[Length[moms], 6] == 0,Return[CHYQMOM4p[moms, ks], Module]]; 
        If[Mod[Length[moms],10] == 0,Return[CHYQMOM9p[moms, ks, q], Module]];   
    ];
    Print["Error: No valid CHYQMOM routine found!"];Abort[];
];
   
CHYQMOM4m[momin_, kk_] := Module[{moms = momin, ks = kk, eqns, vars, linsolv, pm, mymom, n, 
    bu, bv, d20, d11, d02, c20, c11, c02, M1, rho, up, Vf, mu2avg, 
    mu2, M3, rh3, up3, vp21, vp22, rho21, rho22, u, v}, 
    mymom = Pointer[moms, ks]; 

    n = Table[0, {i, 4}]; u = n; v = n; 
    bu = mymom[1, 0]/mymom[0, 0]; 
    bv = mymom[0, 1]/mymom[0, 0]; 
    d20 = mymom[2, 0]/mymom[0, 0]; 
    d11 = mymom[1, 1]/mymom[0, 0]; 
    d02 = mymom[0, 2]/mymom[0, 0]; 
    c20 = d20 - bu^2; 
    c11 = d11 - bu bv;
    c02 = d02 - bv^2; 
    M1 = {1, 0, c20}; 
    {rho, up} = HYQMOM[M1]; 
    Vf = c11 up/c20; 
    mu2avg = c02 - Total[rho Vf^2]; 
    mu2avg = Max[mu2avg, 0]; 
    mu2 = mu2avg; 
    M3 = {1, 0, mu2}; 
    {rh3, up3} = HYQMOM[M3]; 
    vp21 = up3[[1]]; 
    vp22 = up3[[2]]; 
    rho21 = rh3[[1]]; 
    rho22 = rh3[[2]]; 

    n[[1]] = rho[[1]] rho21; 
    n[[2]] = rho[[1]] rho22; 
    n[[3]] = rho[[2]] rho21; 
    n[[4]] = rho[[2]] rho22; 
    n = mymom[0, 0] n; 

    u[[1]] = up[[1]]; 
    u[[2]] = up[[1]]; 
    u[[3]] = up[[2]]; 
    u[[4]] = up[[2]]; 
    u = bu + u; 

    v[[1]] = Vf[[1]] + vp21; 
    v[[2]] = Vf[[1]] + vp22; 
    v[[3]] = Vf[[2]] + vp21; 
    v[[4]] = Vf[[2]] + vp22; 
    v = bv + v; 
    Return[{n,{u,v}}, Module];
];

CHYQMOM4p[momin_,kk_] := Module[{moms = momin, ks = kk, eqns, vars, linsolv, pm, mymom, n, 
    bu, bv, d20, d11, d02, c20, c11, c02, M1, rho, up, Vf, mu2avg, 
    mu2, M3, rh3, up3, vp21, vp22, rho21, rho22, u, v, nro, nn, uu, vv}, 

    nro = Length[DeleteDuplicates[ks[[All,3]]]];
    mymom = Pointer[moms, ks]; 
    (* operating on conditioned moments for each Ro slice *)
    Do[
        n = Table[0, {i, 4}]; u = n; v = n; 
        bu  = mymom[1, 0, l-1]/mymom[0, 0, l-1]; 
        bv  = mymom[0, 1, l-1]/mymom[0, 0, l-1]; 
        d20 = mymom[2, 0, l-1]/mymom[0, 0, l-1]; 
        d11 = mymom[1, 1, l-1]/mymom[0, 0, l-1]; 
        d02 = mymom[0, 2, l-1]/mymom[0, 0, l-1]; 
        c20 = d20 - bu^2; 
        c11 = d11 - bu bv;
        c02 = d02 - bv^2; 
        M1 = {1, 0, c20}; 
        {rho, up} = HYQMOM[M1]; 
        Vf = c11 up/c20; 
        mu2avg = c02 - Total[rho Vf^2]; 
        mu2avg = Max[mu2avg, 0]; 
        mu2 = mu2avg; 
        M3 = {1, 0, mu2}; 
        {rh3, up3} = HYQMOM[M3]; 
        vp21 = up3[[1]]; 
        vp22 = up3[[2]]; 
        rho21 = rh3[[1]]; 
        rho22 = rh3[[2]]; 

        n[[1]] = rho[[1]] rho21; 
        n[[2]] = rho[[1]] rho22; 
        n[[3]] = rho[[2]] rho21; 
        n[[4]] = rho[[2]] rho22; 
        n = mymom[0, 0, l-1] n; 

        u[[1]] = up[[1]]; 
        u[[2]] = up[[1]]; 
        u[[3]] = up[[2]]; 
        u[[4]] = up[[2]]; 
        u = bu + u;

        v[[1]] = Vf[[1]] + vp21; 
        v[[2]] = Vf[[1]] + vp22; 
        v[[3]] = Vf[[2]] + vp21; 
        v[[4]] = Vf[[2]] + vp22; 
        v = bv + v; 

        nn[l] = n;
        uu[l] = u;
        vv[l] = v;
    ,{l,nro}];
    Return[{nn,{uu,vv}}, Module];
];
 

CHYQMOM9m[momin_, kk_, qmax_] := 
  Module[{moms = momin, ks = kk, vars, linsolv, mymom, n, csmall, 
    verysmall, bu, bv, d20, d11, d02, d30, d03, d40, d04, c20, c11, 
    c02, c30, c03, c40, c04, M1, rho, up, Vf, M2, rho2, up2, vp21, 
    vp22, vp23, rho21, rho22, rho23, mu2avg, mu2, mu3, mu4, u, v, 
    eqns, pm, q, eta, slope, det, qm, qp, up3, M3, rh3}, 
    mymom = Pointer[moms, ks]; 
    n = Table[0, {i, 9}]; u = n; v = n; 
    csmall = 10^(-10); 
    verysmall = 10^(-14); 
    bu = mymom[1, 0]/mymom[0, 0]; 
    bv = mymom[0, 1]/mymom[0, 0]; 
    d20 = mymom[2, 0]/mymom[0, 0]; 
    d11 = mymom[1, 1]/mymom[0, 0]; 
    d02 = mymom[0, 2]/mymom[0, 0]; 
    d30 = mymom[3, 0]/mymom[0, 0]; 
    d03 = mymom[0, 3]/mymom[0, 0]; 
    d40 = mymom[4, 0]/mymom[0, 0]; 
    d04 = mymom[0, 4]/mymom[0, 0]; 
    c20 = d20 - bu^2; 
    c11 = d11 - bu bv;
    c02 = d02 - bv^2; 
    c30 = d30 - 3 bu d20 + 2 bu^3; 
    c03 = d03 - 3 bv d02 + 2 bv^3; 
    c40 = d40 - 4 bu d30 + 6 bu^2 d20 - 3 bu^4; 
    c04 = d04 - 4 bv d03 + 6 bv^2 d02 - 3 bv^4; 
    M1 = {1, 0, c20, c30, c40}; 
    {rho, up} = HYQMOM[M1]; 
    If[c20 <= csmall, 
        rho[[1]] = 0; 
        rho[[2]] = 1; 
        rho[[3]] = 0; 
        Vf = 0 up; 
        M2 = {1, 0, c02, c03, c04}; 
        {rho2, up2} = HYQMOM[M2, qmax]; 
        vp21 = up2[[1]]; 
        vp22 = up2[[2]]; 
        vp23 = up2[[3]]; 
        rho21 = rho2[[1]]; 
        rho22 = rho2[[2]]; 
        rho23 = rho2[[3]];
    , 
        Vf = c11 up/c20; 
        mu2avg = c02 - Total[rho Vf^2]; 
        mu2avg = Max[mu2avg, 0]; 
        mu2 = mu2avg; 
        mu3 = 0 mu2; 
        mu4 = mu2^2; 
        If[mu2 > csmall, 
            q = (c03 - Total[rho Vf^3])/mu2^(3/2); 
            eta = (c04 - Total[rho Vf^4] - 6 Total[rho Vf^2] mu2)/mu2^2; 
            If[eta < q^2 + 1, 
                If[Abs[q] > verysmall, 
                    slope = (eta - 3)/q; 
                    det = 8 + slope^2; 
                    qp = 0.5 (slope + Sqrt[det]); 
                    qm = 0.5 (slope - Sqrt[det]); 
                    If[Sign[q] == 1, q = qp, q = qm];
                , 
                    q = 0
                ]; 
                eta = q^2 + 1;
            ]; 
            mu3 = q mu2^(3/2); 
            mu4 = eta mu2^2;
        ]; 
        M3 = {1, 0, mu2, mu3, mu4}; 
        {rh3, up3} = HYQMOM[M3, qmax]; 
        vp21 = up3[[1]]; 
        vp22 = up3[[2]]; 
        vp23 = up3[[3]]; 
        rho21 = rh3[[1]]; 
        rho22 = rh3[[2]]; 
        rho23 = rh3[[3]];
    ]; 

    n[[1]] = rho[[1]] rho21; 
    n[[2]] = rho[[1]] rho22; 
    n[[3]] = rho[[1]] rho23; 
    n[[4]] = rho[[2]] rho21; 
    n[[5]] = rho[[2]] rho22; 
    n[[6]] = rho[[2]] rho23; 
    n[[7]] = rho[[3]] rho21; 
    n[[8]] = rho[[3]] rho22; 
    n[[9]] = rho[[3]] rho23; 
    n = mymom[0, 0] n; 

    u[[1]] = up[[1]]; 
    u[[2]] = up[[1]];
    u[[3]] = up[[1]]; 
    u[[4]] = up[[2]]; 
    u[[5]] = up[[2]];
    u[[6]] = up[[2]];
    u[[7]] = up[[3]]; 
    u[[8]] = up[[3]];
    u[[9]] = up[[3]]; 
    u = bu + u; 

    v[[1]] = Vf[[1]] + vp21; 
    v[[2]] = Vf[[1]] + vp22; 
    v[[3]] = Vf[[1]] + vp23; 
    v[[4]] = Vf[[2]] + vp21; 
    v[[5]] = Vf[[2]] + vp22;
    v[[6]] = Vf[[2]] + vp23; 
    v[[7]] = Vf[[3]] + vp21;
    v[[8]] = Vf[[3]] + vp22; 
    v[[9]] = Vf[[3]] + vp23; 
    v = bv + v; 
    
    Do[
        If[Abs[N[v[[i]]]]<10^(-20),v[[i]]=10.^(-20)];
        If[Abs[N[u[[i]]]]<10^(-20),u[[i]]=10.^(-20)];
    ,{i,9}];
    Return[{n, {u, v}}, Module];
];


CHYQMOM9p[momin_, kk_, qmax_] := 
  Module[{moms = momin, ks = kk, vars, linsolv, mymom, n, csmall, 
    verysmall, bu, bv, d20, d11, d02, d30, d03, d40, d04, c20, c11, 
    c02, c30, c03, c40, c04, M1, rho, up, Vf, M2, rho2, up2, vp21, 
    vp22, vp23, rho21, rho22, rho23, mu2avg, mu2, mu3, mu4, u, v, 
    eqns, pm, q, eta, slope, det, qm, qp, up3, M3, rh3, nro, nn, uu, vv}, 

    nro = Length[DeleteDuplicates[ks[[All,3]]]];
    mymom = Pointer[moms, ks]; 
    Do[
        n = Table[0, {i, 9}]; u = n; v = n; 
        csmall = 10^(-10); 
        verysmall = 10^(-14); 
        bu =  mymom[1, 0,l-1]/mymom[0, 0,l-1]; 
        bv =  mymom[0, 1,l-1]/mymom[0, 0,l-1]; 
        d20 = mymom[2, 0,l-1]/mymom[0, 0,l-1]; 
        d11 = mymom[1, 1,l-1]/mymom[0, 0,l-1]; 
        d02 = mymom[0, 2,l-1]/mymom[0, 0,l-1]; 
        d30 = mymom[3, 0,l-1]/mymom[0, 0,l-1]; 
        d03 = mymom[0, 3,l-1]/mymom[0, 0,l-1]; 
        d40 = mymom[4, 0,l-1]/mymom[0, 0,l-1]; 
        d04 = mymom[0, 4,l-1]/mymom[0, 0,l-1]; 
        c20 = d20 - bu^2; 
        c11 = d11 - bu bv;
        c02 = d02 - bv^2; 
        c30 = d30 - 3 bu d20 + 2 bu^3; 
        c03 = d03 - 3 bv d02 + 2 bv^3; 
        c40 = d40 - 4 bu d30 + 6 bu^2 d20 - 3 bu^4; 
        c04 = d04 - 4 bv d03 + 6 bv^2 d02 - 3 bv^4; 
        M1 = {1, 0, c20, c30, c40}; 
        {rho, up} = HYQMOM[M1]; 
        If[c20 <= csmall, 
            rho[[1]] = 0; 
            rho[[2]] = 1; 
            rho[[3]] = 0; 
            Vf = 0 up; 
            M2 = {1, 0, c02, c03, c04}; 
            {rho2, up2} = HYQMOM[M2, qmax]; 
            vp21 = up2[[1]]; 
            vp22 = up2[[2]]; 
            vp23 = up2[[3]]; 
            rho21 = rho2[[1]]; 
            rho22 = rho2[[2]]; 
            rho23 = rho2[[3]];
        , 
            Vf = c11 up/c20; 
            mu2avg = c02 - Total[rho Vf^2]; 
            mu2avg = Max[mu2avg, 0]; 
            mu2 = mu2avg; 
            mu3 = 0 mu2; 
            mu4 = mu2^2; 
            If[mu2 > csmall, 
                q = (c03 - Total[rho Vf^3])/mu2^(3/2); 
                eta = (c04 - Total[rho Vf^4] - 6 Total[rho Vf^2] mu2)/mu2^2; 
                If[eta < q^2 + 1, 
                    If[Abs[q] > verysmall, 
                        slope = (eta - 3)/q; 
                        det = 8 + slope^2; 
                        qp = 0.5 (slope + Sqrt[det]); 
                        qm = 0.5 (slope - Sqrt[det]); 
                        If[Sign[q] == 1, q = qp, q = qm];
                    , 
                        q = 0
                    ]; 
                    eta = q^2 + 1;
                ]; 
                mu3 = q mu2^(3/2); 
                mu4 = eta mu2^2;
            ]; 
            M3 = {1, 0, mu2, mu3, mu4}; 
            {rh3, up3} = HYQMOM[M3, qmax]; 
            vp21 = up3[[1]]; 
            vp22 = up3[[2]]; 
            vp23 = up3[[3]]; 
            rho21 = rh3[[1]]; 
            rho22 = rh3[[2]]; 
            rho23 = rh3[[3]];
        ]; 

        n[[1]] = rho[[1]] rho21; 
        n[[2]] = rho[[1]] rho22; 
        n[[3]] = rho[[1]] rho23; 
        n[[4]] = rho[[2]] rho21; 
        n[[5]] = rho[[2]] rho22; 
        n[[6]] = rho[[2]] rho23; 
        n[[7]] = rho[[3]] rho21; 
        n[[8]] = rho[[3]] rho22; 
        n[[9]] = rho[[3]] rho23; 
        n = mymom[0, 0, l-1] n; 

        u[[1]] = up[[1]]; 
        u[[2]] = up[[1]];
        u[[3]] = up[[1]]; 
        u[[4]] = up[[2]]; 
        u[[5]] = up[[2]];
        u[[6]] = up[[2]];
        u[[7]] = up[[3]]; 
        u[[8]] = up[[3]];
        u[[9]] = up[[3]]; 
        u = bu + u; 

        v[[1]] = Vf[[1]] + vp21; 
        v[[2]] = Vf[[1]] + vp22; 
        v[[3]] = Vf[[1]] + vp23; 
        v[[4]] = Vf[[2]] + vp21; 
        v[[5]] = Vf[[2]] + vp22;
        v[[6]] = Vf[[2]] + vp23; 
        v[[7]] = Vf[[3]] + vp21;
        v[[8]] = Vf[[3]] + vp22; 
        v[[9]] = Vf[[3]] + vp23; 
        v = bv + v; 

        Do[
            If[Abs[N[v[[i]]]]<10.^(-20),v[[i]]=10^(-20.)];
            If[Abs[N[u[[i]]]]<10.^(-20),u[[i]]=10^(-20.)];
        ,{i,9}];

        nn[l] = n;
        uu[l] = u;
        vv[l] = v;
    ,{l,nro}];
    Return[{nn,{uu,vv}}, Module];
];

HYQMOM[moms_, qmax_: 30] := Module[{}, 
   If[Length[moms] == 3, Return[HYQMOM2[moms], Module]]; 
   If[Length[moms] == 5, Return[HYQMOM3[moms,qmax], Module]]; 
   Print["Error: Unsupported number of moments for HyQMOM."];Abort[];
];
   
HYQMOM2[momin_] := Module[{moms = momin, n, u, bu, d2, c2}, 
    moms = Flatten[moms];
    n = Table[0, {i, 2}]; 
    u = n; 
    bu = moms[[2]]/moms[[1]]; 
    d2 = moms[[3]]/moms[[1]]; 
    c2 = d2 - bu^2; 
    n[[1]] = moms[[1]]/2; 
    n[[2]] = moms[[1]]/2; 
    If[ c2 < 10^(-12), c2 = 10^(-12) ];
    u[[1]] = bu - Sqrt[c2]; 
    u[[2]] = bu + Sqrt[c2]; 
    Return[{n, u}, Module];
];
   
HYQMOM3[momin_, qmax_] := Module[{moms = momin, etasmall, verysmall, 
    realizable, realsmall, n,
    u, bu, d2, d3, d4, c2, c3, c4, q, eta, slope, det, qp, qm, 
    scales, srho, err, mo, scale, dem, prod, rho, ups, up}, 

    moms = Flatten[moms];
    etasmall = 10^(-10); 
    verysmall = 10^(-14); 
    realsmall = 10^(-14); 
    n = Table[0, {i, 3}]; 
    u = n; 
    If[moms[[1]] <= verysmall, 
        n[[2]] = moms[[1]]; 
        Return[{n, u}, Module];
    ]; 
    bu = moms[[2]]/moms[[1]]; 
    d2 = moms[[3]]/moms[[1]]; 
    d3 = moms[[4]]/moms[[1]]; 
    d4 = moms[[5]]/moms[[1]]; 
    c2 = d2 - bu^2; 
    c3 = d3 - 3*bu d2 + 2 bu^3; 
    c4 = d4 - 4*bu d3 + 6 bu^2 d2 - 3 bu^4;
    realizable = c2 c4 - c2^3 - c3^2; 
    If[c2 < 0, 
        If[c2 < -verysmall, Print["Error: c2 negative in three node HYQMOM"];Abort[];]; 
        c2 = 0; 
        c3 = 0; 
        c4 = 0;
    , 
        If[realizable < 0, 
              If[c2 >= etasmall, 
                    q = c3/Sqrt[c2]/c2; 
                    eta = c4/c2/c2; 
                    If[
                        Abs[q] > verysmall, 
                        slope = (eta - 3)/q; 
                        det = 8 + slope^2;
                        qp = 0.5 (slope + Sqrt[det]); 
                        qm = 0.5 (slope - Sqrt[det]); 
                        If[Sign[q] == 1, q = qp, q = qm];
                    , 
                        q = 0;
                    ]; 
                    eta = q^2 + 1; 
                    c3 = q Sqrt[c2] c2; c4 = eta c2^2; 
                    If[realizable < -10^-6,Print["Error: c4 small in HYQMOM3"];Abort[];];
                , 
                    c3 = 0; 
                    c4 = c2^2;
              ];
        ];
    ]; 
    scale = Sqrt[c2]; 
    If[c2 >= etasmall, 
        q = c3/Sqrt[c2]/c2; 
        eta = c4/c2/c2;
    , 
        q = 0; 
        eta = 1;
    ]; 
    If[
        q^2 > qmax^2, 
        slope = (eta - 3)/q; 
        q = qmax Sign[q]; 
        eta = 3 + slope q; 
        realizable = eta - 1 - q^2; 
        If[realizable < 0, eta = 1 + q^2];
    ]; 
    ups[1] = (q - Sqrt[4 eta - 3 q^2])/2; 
    ups[2] = 0; 
    ups[3] = (q + Sqrt[4 eta - 3 q^2])/2; 
    dem = 1/Sqrt[4 eta - 3 q^2]; 
    prod = -ups[1] ups[3]; 
    prod = Max[prod, 1 + realsmall]; 
    rho[1] = -dem/ups[1]; 
    rho[2] = 1 - 1/prod; 
    rho[3] = dem/ups[3]; 
    srho = Sum[rho[i], {i, 3}]; 
    Do[rho[i] = rho[i]/srho, {i, 3}]; 
    scales = Sum[rho[i] ups[i]^2, {i, 3}]/Sum[rho[i], {i, 3}]; 
    Do[up[i] = ups[i] scale/Sqrt[scales], {i, 3}]; 
    If[
        Min[Table[rho[i], {i, 3}]] < 0, 
        Print["Error: Negative weight in HYQMOM"];Abort[];
    ]; 
    n[[1]] = rho[1]; 
    n[[2]] = rho[2]; 
    n[[3]] = rho[3]; 
    n = moms[[1]] n; 

    u[[1]] = up[1]; 
    u[[2]] = up[2]; 
    u[[3]] = up[3]; 
    u = bu + u; 

    mo = Table[Sum[n[[i]] pow[u[[i]], j], {i, 1, 3}], {j, 0, 4}]; 
    err = mo - moms; 
    (* If[err[[3]] > 10^-6 && moms[[1]] > verysmall, Print["err ", err];]; *)
    Return[{n, u}, Module];
];
    

CQMOM[moms_, ks_, perm_] := Module[{},
    If[perm==12,
        If[Length[ks[[1]]] == 2,Return[CQMOM12m[moms, ks], Module]]; 
        If[Length[ks[[1]]] == 3,Return[CQMOM12p[moms, ks], Module]];
    ];
    If[perm==21,
        If[Length[ks[[1]]] == 2,Return[CQMOM21m[moms, ks], Module]]; 
        If[Length[ks[[1]]] == 3,Return[CQMOM21p[moms, ks], Module]];
    ];
    Print["Error: Invalid choice of permutation."];Abort[];
];

CQMOM12m[moms_, ks_]:=Module[{pm, mymom, eqns, vars, linsolv, nr, nrd,
    mRs, xi, w, v, r1, ms, condmoms, condmomvec, mout, xis, ws, wtot,
   wgt,pts,ptsX,ptsY}, 

    nr  = (Max[ks[[All, 1]]] + 1)/2;
    nrd = (Max[ks[[All, 2]]] + 1)/2;
    mymom = Pointer[moms, ks]; 
    mRs=Table[mymom[i,0], {i,0,2nr-1}]; 

    {xi,w}=Wheeler[mRs]; 
    v=Table[Table[xi[[i]]^j,{i,nr}],{j,0,nr-1}]; 
    r1=DiagonalMatrix[w]; 
    Do[ms[i]=Table[mymom[j, i], {j, 0, nr - 1}], {i, 0, 2 nrd - 1}]; 
    Do[condmoms[i]=Re[LinearSolve[v.r1, ms[i]]], {i, 0, 2 nrd - 1}]; 
    Do[
        condmomvec[j]=Table[condmoms[i][[j]], {i, 0, 2 nrd - 1}]; 
        If[Norm[Im[condmomvec[j]]] > 0, Print["Error: Complex conditioned moment vector"];Abort[];];
    , {j, nr}]; 
    Do[{xis[i], ws[i]} = Wheeler[condmomvec[i]], {i,nr}];
    Do[
        wtot[j,i]=w[[j]] ws[j][[i]]; 
        If[
            Re[wtot[j, i]]<0 || Im[wtot[j, i]] != 0, 
            Print["Error: Negative weights"];Abort[];
        ];
    ,{j,nr},{i,nrd}]; 

    wgt = Flatten[Table[wtot[j,k],{j,nr},{k,nrd}]];
    pts = Flatten[Table[Thread[{xi[[i]],xis[i]}],{i,nr}],1];
    ptsX = pts[[All,1]];
    ptsY = pts[[All,2]];

    Return[{wgt,{ptsX,ptsY}}, Module];
];
       
CQMOM12p[moms_, ks_] := Module[{mymom, mRs, xi, w, v, r1, ms, nro, nr, nrd,
	condmoms,condmomvec,xis,ws,wtot,wgt,pts,ptsX,ptsY}, 

    nr  = (Max[ks[[All, 1]]] + 1)/2;
    nrd = (Max[ks[[All, 2]]] + 1)/2;
    nro = Length[DeleteDuplicates[ks[[All,3]]]];
	mymom = Pointer[moms,ks]; 
	Do[
        mRs=Table[mymom[i,0,l-1],{i,0,2nr-1}];
        {xi[l],w}=Wheeler[mRs]; 
        v=Table[Table[xi[l][[i]]^j,{i,nr}],{j,0,nr-1}]; 
        r1=DiagonalMatrix[w];
        Do[ms[m]=Table[mymom[i,m,l-1],{i,0,nr-1}],{m,0,2nrd-1}]; 
        Do[condmoms[i]=LinearSolve[v.r1,ms[i]],{i,0,2 nrd-1}]; 
        Do[
            condmomvec[j]=Table[condmoms[i][[j]],{i,0,2nrd-1}];
            If[Norm[Im[condmomvec[j]]] > 0, Print["Error: Complex conditioned moment vector"];Abort[];];
        ,{j,nr}]; 
        Do[{xis[l][i],ws[i]}=Wheeler[condmomvec[i]],{i,nr}]; 
        Do[wtot[l][j,i]=w[[j]] ws[j][[i]],{i,nrd},{j,nr}];

        wgt[l] = Flatten[Table[wtot[l][j,k],{j,nr},{k,nrd}]];
        pts = Flatten[Table[Thread[{xi[l][[i]],xis[l][i]}],{i,nr}],1];
        ptsX[l] = pts[[All,1]];
        ptsY[l] = pts[[All,2]];
    ,{l,nro}]; 
    Return[{wgt,{ptsX,ptsY}},Module];
];


CQMOM21m[moms_, ks_] := Module[{pm, mymom, eqns, vars, nr, nrd,
    linsolv, mRds, xi, w, v, r1, ms, condmoms, condmomvec, mout, xis, ws, wtot,
    wgt,pts,ptsX,ptsY}, 

    nr  = (Max[ks[[All, 1]]] + 1)/2;
    nrd = (Max[ks[[All, 2]]] + 1)/2;

    mymom = Pointer[moms, ks]; 
    mRds = Table[mymom[0, i], {i, 0, 2 nrd - 1}]; 
    {xi, w} = Wheeler[mRds]; 
    v = Table[Table[xi[[i]]^j, {i, 1, nrd}], {j, 0, nrd - 1}]; 
    r1 = DiagonalMatrix[w]; 
    Do[ms[i] = Table[mymom[i, j], {j, 0, nrd - 1}], {i, 0, 2 nr - 1}]; 
    Do[condmoms[i] = Re[LinearSolve[v.r1, ms[i]]], {i, 0, 2 nr - 1}]; 
    Do[
        condmomvec[j] = Table[condmoms[i][[j]], {i, 0, 2 nr - 1}]; 
        If[Norm[Im[condmomvec[j]]] > 0, Print["Error: Complex conditioned moment vector"];Abort[];];
    , {j, nrd}]; 
    Do[{xis[i], ws[i]} = Wheeler[condmomvec[i]], {i, nrd}];
    Do[wtot[j, i] = w[[j]] ws[j][[i]]; 
        If[Re[wtot[j, i]] < 0 || Im[wtot[j, i]] != 0, 
        Print["Error: Negative weights"];Abort[];];
    ,{j,nrd},{i,nr}]; 


    wgt = Flatten[Table[wtot[j,k],{j,nrd},{k,nr}]];
    pts = Flatten[Table[Thread[{xis[i],xi[[i]]}],{i,nrd}],1];
    ptsX = pts[[All,1]];
    ptsY = pts[[All,2]];

    Return[{wgt,{ptsX,ptsY}}, Module];
];
   

CQMOM21p[moms_, ks_] :=Module[{mymom, mRds, xi, w, v, r1, ms, 
	condmoms,condmomvec,xis,ws,wtot,nro,nr,nrd,wgt,pts,ptsX,ptsY}, 

    nr  = (Max[ks[[All, 1]]] + 1)/2;
    nrd = (Max[ks[[All, 2]]] + 1)/2;
    nro = Length[DeleteDuplicates[ks[[All,3]]]];
	mymom = Pointer[moms,ks]; 
	Do[
        mRds=Table[mymom[0,i,l-1],{i,0,2nrd-1}];
        {xi[l],w}=Wheeler[mRds]; 
        v=Table[Table[xi[l][[i]]^j, {i,nrd}],{j,0,nrd-1}]; 
        r1=DiagonalMatrix[w];
        Do[ms[i]=Table[mymom[i,j,l-1],{j,0,nrd-1}],{i,0,2nr-1}]; 
        Do[condmoms[i]=LinearSolve[v.r1,ms[i]],{i,0,2 nr-1}]; 
        Do[
            condmomvec[j]=Table[condmoms[i][[j]],{i,0,2nr-1}];
            If[Norm[Im[condmomvec[j]]] > 0, Print["Error: Complex conditioned moment vector"];Abort[];];
        ,{j,nrd}]; 
        Do[{xis[l][i],ws[i]}=Wheeler[condmomvec[i]],{i,nrd}]; 
        Do[wtot[l][j,i]=w[[j]] ws[j][[i]],{j,nrd},{i,nr}];

        wgt[l] = Flatten[Table[wtot[l][j,k],{j,nrd},{k,nr}]];
        pts = Flatten[Table[Thread[{xis[l][i],xi[l][[i]]}],{i,nrd}],1];
        ptsX[l] = pts[[All,1]];
        ptsY[l] = pts[[All,2]];
    ,{l,nro}]; 
    Return[{wgt,{ptsX,ptsY}},Module];
];

AdaptiveWheeler[momin_] := Module[
    {eabs,sig,a,b,Ja,w,x,xi,eval,evec,esys,myn,ind,nu,
    nout,cutoff,dab,mab,bmin,z,mindab,maxmab},

    (* rmin is minimum ratio wmin/wmax *)
    (* eabs is minimum distance between distinct abscissas *)

    mom=Flatten[momin];
    nn=Length[mom]/2;
    (* rmin=mins[[1;;nn]] *)
    rmin = Table[0.01,{i,nn}];
    eabs=10^(-4);
    cutoff=0.;

    If[mom[[1]]<0,Print["Error: Negative number density"];Abort[];];
    If[mom[[1]]==0,
        w=0.;
        x=0.;
        Return[{{x},{w}},Module];
    ];
    If[
        nn==1||mom[[1]]<rmin[[1]],
        w=mom[[1]];
        x=mom[[2]]/mom[[1]];
        Return[{{x},{w}},Module];
    ];

    (* Compute modified moments equal to moments *)
    nu=mom;
    (* Construct recurrence matrix *)
    ind=nn;
    a=Table[0.,{i,ind}];
    b=Table[0.,{i,ind}];
    sig=Table[0.,{i,2ind+1},{j,2ind+1}];
    Do[
        sig[[2,i]]=nu[[i-1]];
    ,{i,2,2ind+1}];
    a[[1]] = nu[[2]]/nu[[1]];
    b[[1]] = 0.0;
    Do[
        Do[
            sig[[k,l]] = sig[[k-1,l+1]] - a[[k-2]] sig[[k-1,l]] - b[[k-2]] sig[[k-2,l]];
            If[Abs[sig[[k,l]]] < 10.^(-16),sig[[k,l]] = 10.^(-16)];
        ,{l,k,2*ind - k + 3}];
        a[[k-1]] = sig[[k,k+1]] / sig[[k,k]] - sig[[k-1,k]] / sig[[k-1,k-1]];
        b[[k-1]] = sig[[k,k]] / sig[[k-1,k-1]];
    ,{k,3,ind+1}];
    (* Determine maximum n using diag elements of sig *)
    myn=nn;
    Do[
        If[
            sig[[k,k]]<=cutoff,
            myn=k-2;
            If[
                myn==1,
                w=mom[[1]];
                x=mom[[2]]/mom[[1]];
                Return[{{x},{w}},Module];
            ];
        ];
    ,{k,ind+1,3,-1}];
    (* Compute quadrature using maximum n *)
    a=Table[0,{i,myn}];
    b=Table[0,{i,myn}];
    w=Table[0,{i,myn}];
    x=Table[0,{i,myn}];
    sig=Table[0,{i,2 myn+1},{j,2myn+1}];
    Do[sig[[2,i]]=nu[[i-1]],{i,2,2 myn+1}];
    a[[1]]=nu[[2]]/nu[[1]];
    b[[1]]=0.;
    Do[
        Do[
            sig[[k,l]]=sig[[k-1,l+1]]-a[[k-2]] sig[[k-1,l]]-b[[k-2]] sig[[k-2,l]];
            If[Abs[sig[[k,l]]] < 10.^(-16),sig[[k,l]] = 10.^(-16)];
        ,{l,k,2 myn-k+3}];
        a[[k-1]]=sig[[k,k+1]]/sig[[k,k]]-sig[[k-1,k]]/sig[[k-1,k-1]];
        b[[k-1]]=sig[[k,k]]/sig[[k-1,k-1]];
    ,{k,3,myn+1}];
    (* Check if moments are not realizable (should never happen) *)
    bmin=Min[b];
    If[bmin<0,Print["Error: Moments in Wheeler moments are not realizable"];Abort[];];
    (* Setup Jacobi matrix for n-point quadrature, adapt n using rmax and eabs *)
    Do[
        If[nl==1,
        w=mom[[1]];
        x=mom[[2]]/mom[[1]];
        Return[{{x},{w}},Module];
        ];
        z=Table[0,{i,1,nl},{j,1,nl}];
        Do[
            z[[i,i]]=a[[i]];
            z[[i,i+1]]=Sqrt[b[[i+1]]];
            z[[i+1,i]]=z[[i,i+1]];
        ,{i,1,nl-1}];
        z[[nl,nl]]=a[[nl]];
        (* Compute weights and abscissas *)
        esys=Eigensystem[z];
        eval=esys[[1]];evec=esys[[2]];
        w=Table[0,{i,1,nl}];
        x=w;
        dab=w;
        mab=w;
        x=eval;
        w=Table[evec[[i,1]]^2 mom[[1]],{i,1,nl}];
        Do[
            dab[[i]]=Min[Abs[x[[i]]-x[[1;;i-1]]]];
            mab[[i]]=Max[Abs[x[[i]]-x[[1;;i-1]]]];
        ,{i,nl,2,-1}];
        mindab=Min[dab[[2;;nl]]];
        maxmab=Max[mab[[2;;nl]]];
        If[nl==2,maxmab=1];
        (* Check conditions that weights and abscissas must both satisfy *)
        If[
            Min[w]/Max[w]>rmin[[nl]]&&mindab/maxmab>eabs,
            Return[{x,w},Module];
        ];
    ,{nl,myn,1,-1}];
];


VanderSolve[x_,q_]:=Module[{n,w,b,s,t,xx,c},
    n=Length[q];
    If[n==1,
        w={q[[1]]};
        Return[w,Module];
    ];
    c=Table[0,{i,n}];
    w=c;
    Do[
        xx=-x[[i]];
        Do[
            c[[j]]=c[[j]]+xx c[[j+1]];
        ,{j,n+1-i,n-1}];
        c[[n]]=c[[n]]+xx;
    ,{i,2,n}];
    Do[
        xx=x[[i]];
        t=1;
        b=1;
        s=q[[n]];
        Do[
            b=c[[k]]+xx b;
            s=s+q[[k-1]]b;
            t=xx t+b;
        ,{k,n,2,-1}];
        w[[i]]=s/t;
    ,{i,1,n}];
    Return[w,Module];
];

err[fine_,coarse_]:=Norm[fine-coarse]/Norm[fine];

RK23[mom_,myrhs_,t_,dt_]:=Module[{moms=mom,mome,momstemp1,momstemp2,rhs},
	(* SSP-RK2 *)
	{moms,rhs}=myrhs[moms,t];
	momstemp1=moms+dt rhs;
	{momstemp1,rhs}=myrhs[momstemp1,t+dt];
	mome=(1/2) moms+(1/2)(momstemp1+dt rhs);

	(* SSP-RK3 *)
	momstemp2=(3/4)moms+(1/4)(momstemp1+dt rhs);
	{momstemp2,rhs}=myrhs[momstemp2,t+dt/2];
	moms=(1/3)moms+(2/3)(momstemp2+dt rhs);
	Return[{moms,err[moms,mome]},Module];
];




End[];
EndPackage[];

(* 
Transportmerms[eqn_,minvars_]:=Module[
    {v,integrand,dim,mrdd,list,exp,coefs,exps,vars,l,m,n},

    invars = minvars[[1]];
    v4 = minvars[[2]];
    dim=Length[invars];

    idx=Table[Symbol["c"][i],{i,dim}];

    If[dim == 1, 
        vars = {v[1]}; 
        {l} = idx;
    ];
    If[dim==2,
        vars={v[1],v[2]};
        {l,m}=idx;
    ];
    If[dim==3,
        vars={v[1],v[2],v[3]};
        {l,m,n}=idx;
    ];
    If[dim!=1&&dim!=2&&dim!=3,
        Print["Error: Dimesionallity not supported. You used dim = ",dim];Abort[];
    ];

    Do[v[i]=invars[[i]],{i,dim}];
    mrdd=v4/.Solve[eqn,v4][[1]];

    If[dim==1,integrand=mrdd v[1]^(l-1)];
    If[dim==2,integrand=mrdd v[1]^l v[2]^(m-1)];
    If[dim==3,integrand=mrdd v[1]^l v[2]^(m-1) v[3]^n];

    list=PowerExpand[List@@Distribute[integrand]];
    exps=Table[Exponent[list[[i]],j],{i,Length[list]},{j,invars}];

    If[dim==1,
        coefs=l Table[Coefficient[list[[i]],Product[v[j]^exps[[i,j]],{j,dim}]],{i,Length[list]}];
    ,
        coefs=m Table[Coefficient[list[[i]],Product[v[j]^exps[[i,j]],{j,dim}]],{i,Length[list]}];
    ];

    (* Add term that isn't in the integrand *)
    If[dim==2,AppendTo[exps,{l-1,m+1}]];
    If[dim==3,AppendTo[exps,{l-1,m+1,n}]];
    If[dim>1,AppendTo[coefs,l]];
    Return[{coefs,exps},Module];
];
*)
