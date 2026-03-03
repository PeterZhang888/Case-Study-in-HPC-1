"""
Q2 GMRES with Progressive Givens Rotations

Algorithm Steps:

1. Compute $r_0 = b - A x_0$, $\beta = \|r_0\|_2$, and $v_1 = r_0 / \beta$.

   Set $g = \beta e_1$.

2. For $j = 1, 2, ..., m$ do:

3. &nbsp;&nbsp;&nbsp;&nbsp; Compute $w_j = A v_j$.

4. &nbsp;&nbsp;&nbsp;&nbsp; For $i = 1, ..., j$ do:

5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $h_{ij} = (w_j, v_i)$.

6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w_j = w_j - h_{ij} v_i$.

7. &nbsp;&nbsp;&nbsp;&nbsp; End For.

8. &nbsp;&nbsp;&nbsp;&nbsp; $h_{j+1,j} = \|w_j\|_2$. If $h_{j+1,j} = 0$, set $m = j$ and go to Step 10 then 14.

9. &nbsp;&nbsp;&nbsp;&nbsp; $v_{j+1} = w_j / h_{j+1,j}$.

10. &nbsp;&nbsp;&nbsp;&nbsp; Apply all previously computed Givens rotations $\Omega_1, ..., \Omega_{j-1}$
    to the new column of $\bar{H}_m$.

11. &nbsp;&nbsp;&nbsp;&nbsp; Compute a new Givens rotation $\Omega_j$ that annihilates $h_{j+1,j}$,
    making the entry zero and preserving the 2-norm.

12. &nbsp;&nbsp;&nbsp;&nbsp; Apply the same rotation $\Omega_j$ to the vector $g$.
    The residual norm becomes $\|r_j\|_2 = |g_{j+1}|$.

13. End For.

14. Solve the upper triangular system $R_m y_m = g_{1:m}$ by back substitution.

15. Set $x_m = x_0 + V_m y_m$.
"""

import numpy as np

def gmres_givens(A, b, x0, m):
    """
    GMRES using progressive Givens rotations.
    """
    n= len(b)
    r0= b - A @ x0
    beta = np.linalg.norm(r0)
    if beta == 0:
        return x0, 0.0
    V= np.zeros((n, m+1))
    H= np.zeros((m+1, m))
    # Givens storage
    c= np.zeros(m)
    s= np.zeros(m)

    # RHS vector (beta e1)
    g= np.zeros(m+1)
    g[0]= beta
    V[:, 0]= r0 / beta
    j_final=-1
    residual=0
    for j in range(m):
        #Arnoldi
        w= A @ V[:, j]
        for i in range(j+1):
            H[i, j]= np.dot(V[:, i], w)
            w-= H[i, j] * V[:, i]
        H[j+1, j]= np.linalg.norm(w)
        if H[j+1, j] != 0:
            V[:, j+1]= w / H[j+1, j]
        # apply previous Givens rotations
        for i in range(j):
            H_temp= c[i]*H[i, j] + s[i]*H[i+1, j]
            H[i+1, j]= -s[i]*H[i, j] + c[i]*H[i+1, j]
            H[i, j]=H_temp
        if H[j+1, j] == 0:
            # No new rotation
            residual = abs(g[j])
            j_final = j
            break
        
        # compute new Givens rotation
        denom= np.hypot(H[j, j], H[j+1, j])
        c[j]= H[j, j] / denom
        s[j]= H[j+1, j] / denom
        # apply new rotation to H
        H[j, j]= c[j]*H[j, j] + s[j]*H[j+1, j]
        H[j+1, j]= 0.0
        # apply same rotation to g
        g_temp= c[j]*g[j] + s[j]*g[j+1]
        g[j+1]= -s[j]*g[j] + c[j]*g[j+1]
        g[j]=g_temp

        # residual norm
        residual = abs(g[j+1])
        j_final=j
    # solve upper triangular system
    y = np.zeros(j_final+1)

    for i in range(j_final, -1, -1):
        y[i]= (g[i] - H[i, i+1:j_final+1] @ y[i+1:j_final+1]) / H[i, i]
    x = x0 + V[:, :j_final+1] @ y
    return x, residual