"""autowind: automatic relative-winding chains (villa relative_windings.json) from a surface mask.

A packaging layer over tools/auto_windings.py (unchanged); see release/AUTOWIND.md.
"""

import os

# Same as tools/auto_windings.py, but before anything imports numpy: single-threaded BLAS makes the patch-graph solve
# bit-reproducible (multi-threaded OpenBLAS changes the summation order -> last-ulp differences in the windings).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

__version__ = "0.1.0"
