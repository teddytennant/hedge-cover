# hedge-cover

Hedge over a finite cover of selector-consistent hypotheses for smoothed online regression, in JAX on CPU.

pip install -e .

JAX_PLATFORMS=cpu python -m pytest

The paper's selector is an information-theoretic well-ordering of an infinite class, and the authors defer computation. This package evaluates a finite list of functions and enumerates all (m+1)^T hypotheses, so it only runs for small T (the default m=2T exceeds the cap past T=4). The grid is -1 + 2j/m for j=0..m-1.
