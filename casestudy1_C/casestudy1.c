/*
 * casestudy1.c : Parallel TSQR with Four MPI Processes.

 * Each rank owns m/4 rows of A.
 * Uses LAPACKE_dgeqrf + LAPACKE_dorgqr for local QR.
 * Reduction tree: rank(0,1)-> rank 0 ; rank(2,3)-> rank 2 ; then rank (0,2)-> rank 0.
 * Produces:
      * R_final on rank 0 (n x n, upper triangular)
      * Q_local_final on every rank (mloc x n)
 * Test: rank 0 gathers Q blocks, forms QR, computes ||A-QR||_F / ||A||_F.
*/
#include "mpi.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "mkl_lapacke.h"
#include "mkl_cblas.h"

void TSQR(const double *W_i, int mloc, int n,double *Q_i_final, double *R_root_out,MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "TSQR requires exactly 4 ranks in the provided communicator (got %d).\n", size);
        }
        MPI_Abort(comm, 1);
    }
    // 1) Local QR: W_i = Q_i * R_i
    double *W_copy = (double*)malloc((size_t)mloc*(size_t)n*sizeof(double));
    double *tau   = (double*)malloc((size_t)n*sizeof(double));
    double *R_i   = (double*)calloc((size_t)n*(size_t)n, sizeof(double));
    if (!W_copy || !tau || !R_i) {
        fprintf(stderr, "Rank %d: malloc failed in local QR.\n", rank);
        MPI_Abort(comm, 1);
    }
    memcpy(W_copy, W_i, (size_t)mloc*(size_t)n*sizeof(double));
    // QR factorisation with Householder method
    int sign=LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, mloc, n, W_copy, n, tau);
    if (sign!= 0) {
        fprintf(stderr, "Rank %d: dgeqrf failed (sign=%d).\n", rank, sign);
        MPI_Abort(comm, 1);
    }
    // Extract R_i from upper triangle (first n rows)
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            R_i[i*n + j] = W_copy[i*n + j];
        }
    }
    // Build explicit Q_i into Q_i_final
    memcpy(Q_i_final, W_copy, (size_t)mloc*(size_t)n*sizeof(double));
    sign = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, mloc, n, n, Q_i_final, n, tau);
    if (sign != 0) {
        fprintf(stderr, "Rank %d: dorgqr failed (sign=%d).\n", rank, sign);
        MPI_Abort(comm, 1);
    }
    free(W_copy);
    free(tau);
    double *X_i= (double*)calloc((size_t)n*(size_t)n, sizeof(double));
    double *Y= (double*)calloc((size_t)n*(size_t)n, sizeof(double));
    if (!X_i || !Y) {
        fprintf(stderr, "Rank %d: malloc failed for X_i/Y.\n", rank);
        MPI_Abort(comm, 1);
    }
    
    // 2) Level-1 reduction: rank (0,1)-> rank 0 and rank (2,3)-> rank 2
    double *R_pair = NULL;
    int partner=-1;
    if (rank==0) partner=1;
    if (rank==2) partner=3;
    if (rank == 0 || rank == 2){
        double *R_partner = (double*)malloc((size_t)n*(size_t)n*sizeof(double));
        if (!R_partner) {
            fprintf(stderr, "Rank %d: malloc failed for R_partner.\n", rank);
            MPI_Abort(comm, 1);
        }
        MPI_Recv(R_partner, n*n, MPI_DOUBLE, partner, 100, comm, MPI_STATUS_IGNORE);
        double *R_stack1   = (double*)malloc((size_t)(2*n)*(size_t)n*sizeof(double));
        double *tau_1 = (double*)malloc((size_t)n*sizeof(double));
        if (!R_stack1 || !tau_1) {
            fprintf(stderr, "Rank %d: malloc failed in level-1 reduction.\n", rank);
            MPI_Abort(comm, 1);
        }
    // Stack [R_i; R_partner]
        memcpy(R_stack1, R_i, (size_t)n*(size_t)n*sizeof(double));
        memcpy(R_stack1 + (size_t)n*(size_t)n, R_partner, (size_t)n*(size_t)n*sizeof(double));

        sign = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, 2*n, n, R_stack1, n, tau_1);
        if (sign != 0) {
            fprintf(stderr, "Rank %d: level-1 dgeqrf failed (info=%d).\n", rank, sign);
            MPI_Abort(comm, 1);
        }
    
        R_pair = (double*)calloc((size_t)n*(size_t)n, sizeof(double));
        if (!R_pair) {
            fprintf(stderr, "Rank %d: calloc failed for R_pair.\n", rank);
            MPI_Abort(comm, 1);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                R_pair[i*n + j] = R_stack1[i*n + j];
            }
        }
    
        sign = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, 2*n, n, n, R_stack1, n, tau_1);
        if (sign != 0) {
            fprintf(stderr, "Rank %d: level-1 dorgqr failed (info=%d).\n", rank, sign);
            MPI_Abort(comm, 1);
        }
    // Qhat = [X_self; X_partner]
        memcpy(X_i, R_stack1, (size_t)n*(size_t)n*sizeof(double));
        double *X_partner = (double*)malloc((size_t)n*(size_t)n*sizeof(double));
        if (!X_partner) {
            fprintf(stderr, "Rank %d: malloc failed for S_partner.\n", rank);
            MPI_Abort(comm, 1);
        }
        memcpy(X_partner, R_stack1 + (size_t)n*(size_t)n, (size_t)n*(size_t)n*sizeof(double));
        MPI_Send(X_partner, n*n, MPI_DOUBLE, partner, 101, comm);

        free(X_partner);
        free(R_partner);
        free(R_stack1);
        free(tau_1);
    } else {
        int reducer = (rank == 1) ? 0 : 2;
        MPI_Send(R_i, n*n, MPI_DOUBLE, reducer, 100, comm);
        MPI_Recv(X_i, n*n, MPI_DOUBLE, reducer, 101, comm, MPI_STATUS_IGNORE);
    }
    
    // 3) Level-2 reduction: rank (0,2)-> rank 0
    if (rank == 0) {
        double *R11 = (double*)malloc((size_t)n*(size_t)n*sizeof(double));
        if (!R11) {
            fprintf(stderr, "Rank 0: malloc failed for R11.\n");
            MPI_Abort(comm, 1);
        }
        MPI_Recv(R11, n*n, MPI_DOUBLE, 2, 200, comm, MPI_STATUS_IGNORE);
        double *R_stack2 = (double*)malloc((size_t)(2*n)*(size_t)n*sizeof(double));
        double *tau_2 = (double*)malloc((size_t)n*sizeof(double));
        if (!R_stack2 || !tau_2) {
            fprintf(stderr, "Rank 0: malloc failed in level-2 reduction.\n");
            MPI_Abort(comm, 1);
        }
        memcpy(R_stack2, R_pair, (size_t)n*(size_t)n*sizeof(double));
        memcpy(R_stack2 + (size_t)n*(size_t)n, R11, (size_t)n*(size_t)n*sizeof(double));
        sign = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, 2*n, n, R_stack2, n, tau_2);
        if (sign != 0) {
            fprintf(stderr, "Rank 0: level-2 dgeqrf failed (sign=%d).\n", sign);
            MPI_Abort(comm, 1);
        }
        for (int i = 0; i < n*n; i++) R_root_out[i] = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                R_root_out[i*n + j] = R_stack2[i*n + j];
            }
        }
        sign = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, 2*n, n, n, R_stack2, n, tau_2);
        if (sign != 0) {
            fprintf(stderr, "Rank 0: level-2 dorgqr failed (sign=%d).\n", sign);
            MPI_Abort(comm, 1);
        }

        // Y0 (top) stays here and is sent to rank 1
        memcpy(Y, R_stack2, (size_t)n*(size_t)n*sizeof(double));
        MPI_Send(Y, n*n, MPI_DOUBLE, 1, 202, comm);

        // Y1 (bottom) sent to rank 2
        double *Y1 = (double*)malloc((size_t)n*(size_t)n*sizeof(double));
        if (!Y1) {
            fprintf(stderr, "Rank 0: malloc failed for Y1.\n");
            MPI_Abort(comm, 1);
        }
        memcpy(Y1, R_stack2 + (size_t)n*(size_t)n, (size_t)n*(size_t)n*sizeof(double));
        MPI_Send(Y1, n*n, MPI_DOUBLE, 2, 201, comm);

        free(Y1);
        free(R11);
        free(R_stack2);
        free(tau_2);
    }
    else if (rank == 2) {
        MPI_Send(R_pair, n*n, MPI_DOUBLE, 0, 200, comm);
        MPI_Recv(Y, n*n, MPI_DOUBLE, 0, 201, comm, MPI_STATUS_IGNORE);
        MPI_Send(Y, n*n, MPI_DOUBLE, 3, 203, comm);
    }
    else if (rank == 1) {
        MPI_Recv(Y, n*n, MPI_DOUBLE, 0, 202, comm, MPI_STATUS_IGNORE);
    }
    else if (rank == 3) {
        MPI_Recv(Y, n*n, MPI_DOUBLE, 2, 203, comm, MPI_STATUS_IGNORE);
    }
    
    
    // 4) Q_i_final = Q_i * (X_i * Y)
    double *XY=(double*)calloc((size_t)n*(size_t)n, sizeof(double));
    double *Q_i=(double*)malloc((size_t)mloc*(size_t)n*sizeof(double));
    if (!XY || !Q_i) {
        fprintf(stderr, "Rank %d: malloc failed for XY/Q_i.\n", rank);
        MPI_Abort(comm, 1);
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,n, n, n, 1.0, X_i, n, Y, n, 0.0, XY, n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,mloc, n, n, 1.0, Q_i_final, n, XY, n, 0.0, Q_i, n);
    memcpy(Q_i_final, Q_i, (size_t)mloc*(size_t)n*sizeof(double));

    // cleanup
    free(Q_i);
    free(XY);
    free(X_i);
    free(Y);
    free(R_i);
    if (R_pair) free(R_pair);
}




int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) {
             fprintf(stderr, "This TSQR case study requires exactly 4 MPI ranks (got %d).\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Set the size of matrix
    const int m = 400;
    const int n = 10;
    const int mloc = m / 4;
    // Allocate local W_i (mloc x n)
    double *W_i = (double*)malloc((size_t)mloc*(size_t)n*sizeof(double));
    double *W_i_copy = (double*)malloc((size_t)mloc*(size_t)n*sizeof(double));
    double *Q_i = (double*)malloc((size_t)mloc*(size_t)n*sizeof(double));
    double *R_root = (double*)malloc((size_t)n*(size_t)n*sizeof(double));

    if (!W_i || !W_i_copy || !Q_i || !R_root) {
        fprintf(stderr, "Rank %d: malloc failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Fill W_i with values
    // global row index for each local row:
    srand(1234 + rank);
    for (int i = 0; i < mloc; i++) {
        for (int j = 0; j < n; j++) {
            int r = rand() % 21 - 10;   // integer in [-10, 10]
             W_i[i*n + j] = (double) r;
        }
    }
    
     memcpy(W_i_copy, W_i, (size_t)mloc*(size_t)n*sizeof(double));
    // Call TSQR
    // After this:
    //    Q_i contains the local block of the final Q (mloc x n)
    //    R_root contains final R (n x n) on rank 0;
    TSQR(W_i, mloc, n, Q_i, R_root, MPI_COMM_WORLD);
    // Correctness check: ||W - Q R||_F / ||W||_F
    // Do it on rank 0 by gathering W and Q
    double *W = NULL;
    double *Q = NULL;
    if (rank == 0) {
        W = (double*)malloc((size_t)m*(size_t)n*sizeof(double));
        Q = (double*)malloc((size_t)m*(size_t)n*sizeof(double));
        if (!W || !Q) {
            fprintf(stderr, "Rank 0: malloc failed for W/Q.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Gather(W_i_copy, mloc*n, MPI_DOUBLE, W, mloc*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(Q_i,      mloc*n, MPI_DOUBLE, Q, mloc*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Compute QR = Q * R_root
        double *QR = (double*)malloc((size_t)m*(size_t)n*sizeof(double));
        if (!QR) {
            fprintf(stderr, "Rank 0: malloc failed for QR.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1.0,Q, n,R_root, n,0.0,QR, n);
        // Frobenius norm of residual and W
        double num = 0.0, den = 0.0;
        for (int k = 0; k < m*n; k++) {
            double r = W[k] - QR[k];
            num += r*r;
            den += W[k]*W[k];
        }
        num = sqrt(num);
        den = sqrt(den);
        printf("Residual ||W - QR||_F / ||W||_F = %.6e\n", num / den);
        free(QR);
        free(W);
        free(Q);
    }

    free(W_i);
    free(W_i_copy);
    free(Q_i);
    free(R_root);

    MPI_Finalize();
    return 0;
}



















