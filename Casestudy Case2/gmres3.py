#Q3 GMRES using progressive Givens rotations with residuals tracking

import numpy as np

def gmres_givens_R(A, b, x0, m, tol):
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
    residuals = []
    for j in range(m):
        # Arnoldi
        w= A @ V[:, j]
        for i in range(j+1):
            H[i, j]= np.dot(V[:, i], w)
            w-= H[i, j] * V[:, i]
        H[j+1, j]= np.linalg.norm(w)
        if H[j+1, j] != 0:
            V[:, j+1]= w / H[j+1, j]
        #apply previous Givens rotations
        for i in range(j):
            H_temp= c[i]*H[i, j] + s[i]*H[i+1, j]
            H[i+1, j]= -s[i]*H[i, j] + c[i]*H[i+1, j]
            H[i, j]=H_temp
        if H[j+1, j] == 0:
            #no new rotation needed
            residual = abs(g[j])
            j_final = j
            break
        
        #compute new Givens rotation
        denom= np.hypot(H[j, j], H[j+1, j])
        c[j]= H[j, j] / denom
        s[j]= H[j+1, j] / denom
        #apply new rotation to H
        H[j, j]= c[j]*H[j, j] + s[j]*H[j+1, j]
        H[j+1, j]= 0.0
        #apply same rotation to g
        g_temp= c[j]*g[j] + s[j]*g[j+1]
        g[j+1]= -s[j]*g[j] + c[j]*g[j+1]
        g[j]=g_temp

        #residual norm
        residual = abs(g[j+1])
        residuals.append(residual)
        j_final=j
        #track the residual
        if residual < tol:
            break

    #solve upper triangular system
    y = np.zeros(j_final+1)

    for i in range(j_final, -1, -1):
        y[i]= (g[i] - H[i, i+1:j_final+1] @ y[i+1:j_final+1]) / H[i, i]
    x = x0 + V[:, :j_final+1] @ y
    return x, residuals