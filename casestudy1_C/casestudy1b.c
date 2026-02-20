/**
 * casestudy1b.c : Implement TSQR on different m and n dimentions
 * The purpose of the code is to study how runtime scales with:
 *
 *   1) Increasing number of rows (m) while keeping n fixed. (with warmup process before the experiments)
 *   2) Increasing number of columns (n) while keeping m fixed.
 *
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

    // 2) Level-1 reduction:rank (0,1)->rank 0 and rank (2,3)->rank 2
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "Need exactly 4 MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    //The first call to MKL's LAPACK routines incurs one-time initialization costs like kernel selection and thread spawning that pollute our timing measurements at the very beginning, so we use a warmup call to absorb these costs before we start measuring.
    {
           int nw = 50, mw = nw * 10;
           double *Aw = malloc(mw * nw * sizeof(double));
           double *Qw = malloc(mw * nw * sizeof(double));
           double *Rw = malloc(nw * nw * sizeof(double));
           srand48(42);
           for (int j = 0; j < mw * nw; j++) Aw[j] = drand48();
           TSQR(Aw, mw, nw, Qw, Rw, MPI_COMM_WORLD);
           free(Aw); free(Qw); free(Rw);
           MPI_Barrier(MPI_COMM_WORLD);
    }
    // Test 1: Fixed n, varying m
    int n_fixed = 50;
    int m_values[] = {200, 2000, 20000, 200000, 2000000};
    
    for (int i = 0; i < 5; i++) {
        int m = m_values[i];
        int n = n_fixed;
        int mloc = m / 4;  // Rows per processor
        
        // Generate random LOCAL portion for each rank
        double *A_local = malloc(mloc * n * sizeof(double));
        srand48(1234 + rank + i*1000+m);  // Different seed per rank
        for (int j = 0; j < mloc * n; j++) {
            A_local[j] = -100.0 + 200.0 * drand48();;
        }
        
        // Allocate LOCAL output arrays (mloc × n per rank)
        double *Q_local = malloc(mloc * n * sizeof(double));
        double *R_root = malloc(n * n * sizeof(double));
        
        // Time measurement
	int NREPS = 5;
        double total = 0.0;
        for (int rep = 0; rep < NREPS; rep++) {
                MPI_Barrier(MPI_COMM_WORLD);
                double start=MPI_Wtime();
                TSQR(A_local, mloc, n, Q_local, R_root, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		double end=MPI_Wtime();
                double elapsed= end - start;
		double global_elapsed;
		MPI_Reduce(&elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		if (rank==0) total+=global_elapsed;
        }
        if (rank == 0)
                printf("m=%d, n=%d, avg_time=%.6f s\n", m, n, total / NREPS);
        
        free(A_local);
        free(Q_local);
        free(R_root);
    }
    
    // Test 2: Fixed m, varying n
    int n_values[] = {5, 10, 50, 100,500,1000};
    int m_fixed = 800000;
    
    for (int i = 0; i < 6; i++) {
        int m = m_fixed;
        int n = n_values[i];
        int mloc = m / 4;
        
        double *A_local = malloc(mloc * n * sizeof(double));
        srand48(5678 + rank + i*1000+n);
        for (int j = 0; j < mloc * n; j++) {
            A_local[j] = -100.0 + 200.0 * drand48();
        }
        
        double *Q_local = malloc(mloc * n * sizeof(double));
        double *R_root = malloc(n * n * sizeof(double));
        
        int NREPS = 5;
        double total = 0.0;
        for (int rep = 0; rep < NREPS; rep++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double start = MPI_Wtime();          
            TSQR(A_local, mloc, n, Q_local, R_root, MPI_COMM_WORLD);
            double end = MPI_Wtime();           
            total += (end - start);
        }

        if (rank == 0)
            printf("m=%d, n=%d, avg_time=%.6f s\n", m, n, total / NREPS);
        free(A_local);
        free(Q_local);
        free(R_root);
    }
    
    MPI_Finalize();
    return 0;
}
