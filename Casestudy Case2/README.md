# GMRES Sparse Matrix Experiments

## Repository Structure

| File | Description |
|------|-------------|
| `gmres1.py` | Basic GMRES implementation using least-squares |
| `gmres2.py` | GMRES implementation using Givens rotations |
| `gmres3.py` | GMRES implementation using Givens rotations with residual tracking |
| `casestudy2.ipynb` | Experiments, results, and analysis |

## Usage

All function definitions are contained in the `.py` files and imported into the notebook:

```python
from gmres1 import gmres
from gmres2 import gmres_givens
from gmres3 import gmres_givens_R
```

Open `casestudy2.ipynb` to run the experiments and view the results.
