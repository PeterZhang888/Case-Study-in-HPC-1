"""
Q1: GMRES Algorithm

Algorithm Steps:

1. Compute $r_0 = b - Ax_0$, $\beta = \|r_0\|_2$, and $v_1 = r_0 / \beta$

2. For $j = 1, 2, ..., m$ do:

3. &nbsp;&nbsp;&nbsp;&nbsp; Compute $w_j = Av_j$

4. &nbsp;&nbsp;&nbsp;&nbsp; For $i = 1, ..., j$ do:

5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $h_{ij} = (w_j, v_i)$

6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w_j = w_j - h_{ij}v_i$

7. &nbsp;&nbsp;&nbsp;&nbsp; End For

8. &nbsp;&nbsp;&nbsp;&nbsp; $h_{j+1,j} = \|w_j\|_2$. If $h_{j+1,j} = 0$, set $m = j$ and go to Step 11

9. &nbsp;&nbsp;&nbsp;&nbsp; $v_{j+1} = w_j / h_{j+1,j}$

10. End For

11. Define the $(m + 1) \times m$ Hessenberg matrix $\bar{H}_m = \{h_{ij}\}$ for $1 \leq i \leq m+1$ and $1 \leq j \leq m$

12. Compute $y_m$ as the minimizer of $\|\beta e_1 - \bar{H}_m y\|_2$ and set $x_m = x_0 + V_m y_m$
"""

import numpy as np

def gmres(A, b, x0, m):
    """
    Parameters:
    A : ndarray (n x n)
    b : ndarray (n,)
    x0 : initial guess (n,)
    m : number of Krylov iterations
    
    Returns:
    x_m : approximate solution
    residual_norm : ||b - A x_m||
    """
    n=len(b)
    # initial residual
    r0=b - A @ x0
    beta= np.linalg.norm(r0)
    if beta == 0:
        return x0, 0.0

    # allocate storage
    V= np.zeros((n, m+1))
    H= np.zeros((m+1, m))

    # first basis vector
    V[:, 0]= r0 / beta
    #Arnoldi process
    for j in range(m):
        w= A @ V[:, j]
        #Modified Gram-Schmidt
        for i in range(j+1):
            H[i, j]= np.dot(V[:, i], w)
            w= w - H[i, j] * V[:, i]
        H[j+1, j]= np.linalg.norm(w)
        if H[j+1, j]== 0:
            #breakdown
            m= j + 1
            break
        V[:, j+1]= w / H[j+1, j]

    # build right-hand side vector beta e1
    e1= np.zeros(m+1)
    e1[0]= beta

    # solve least squares problem
    y, _, _, _ = np.linalg.lstsq(H[:m+1, :m], e1, rcond=None)
    # solution
    x_m= x0 + V[:, :m] @ y
    # Compute final residual
    residual_norm = np.linalg.norm(b - A @ x_m)
    return x_m, residual_norm