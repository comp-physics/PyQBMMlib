
import * 

# Similar to MomentIndices in Mathematica
#
# Input: Number of quadrature points/nodes, 
#       inversion method, number of permutations
#
# Output: Required/carried moment indices (indices)
#
# Notes: Number of perturmations required (for 2/3D)
#       so we know which moments are needed for inversion 
#
#       nr  = number of quadrature points in 1st coord dir
#       nrd = number of quadrature points in 2nd coord dir
#       nro -> ignore!

If[method == "QMOM" || method == "AQMOM",
indices = Table[{i},{i,0,2 nr - 1}]; 
];

If[method == "HYQMOM",
If[nr==2,indices = {{0},{1},{2}}];
If[nr==3,indices = {{0},{1},{2},{3},{4}}];
];


If[method == "CHYQMOM", 
If[nr == 2, 
If[
nro == 0, 
indices = {{0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {0, 2}};
,
indices = Flatten[Table[{{0, 0, i}, {1, 0, i}, {0, 1, i}, {2, 0, i}, {1, 1, i}, {0, 2, i}}, {i, 0, nro-1}], 1];
];
]; 
If[nr == 3, 
If[
nro == 0, 
indices = {{0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {0, 2}, {3, 0}, {0,3}, {4, 0}, {0, 4}}
,
indices = Flatten[Table[{{0, 0, i}, {1, 0, i}, {0, 1, i}, {2, 0, i}, {1, 1, i}, {0, 2, i}, {3, 0, i}, {0,3, i}, {4, 0, i}, {0, 4, i}}, {i,0,nro-1}],1];
];
]; 
]; 


If[method == "CQMOM", 
If[nro == 0, 
k1 = Flatten[{Table[{q,p}, {q, 0, nr - 1}, {p, 0, 2 nrd - 1}], Table[{q, p}, {q, nr, 2 nr - 1}, {p, 0, 0}]}, 2]; 
k2 = Flatten[{Table[{p,q}, {q, 0, nrd - 1}, {p, 0, 2 nr - 1}], Table[{p, q}, {q, nrd, 2 nrd - 1}, {p, 0, 0}]}, 2]; 
If[
numperm==2,
indices = DeleteDuplicates[Join[k1, k2]]; 
,
indices = DeleteDuplicates[k1]; 
];
,
Do[
k1[i] = Flatten[{Table[{q,p,i}, {q, 0, nr - 1}, {p, 0, 2 nrd - 1}], Table[{q,p,i}, {q, nr, 2 nr - 1}, {p, 0, 0}]}, 2]; 
k2[i] = Flatten[{Table[{p,q,i}, {q, 0, nrd - 1}, {p, 0, 2 nr - 1}], Table[{p,q,i}, {q, nrd, 2 nrd - 1}, {p, 0, 0}]}, 2]; 
If[
numperm==2,
indicestemp[i] = DeleteDuplicates[Join[k1[i],k2[i]]]; 
,
indicestemp[i] = DeleteDuplicates[k1[i]]; 
];
,{i,0,nro-1}];
indices=Flatten[Join[Table[indicestemp[i],{i,0,nro-1}]],1];
];
]; 

return indices
