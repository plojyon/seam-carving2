#include <omp.h>
#include <stdio.h>

#define NUM_THREADS 4

#define KILO (1000)
#define MEGA (1000*KILO)
#define GIGA (1000*MEGA)

#define NUM_STEPS GIGA

int main () {
    double pi, sum = 0.0;
    int sanity = 0;
    const double step = 1.0 / MEGA;
    printf("Running program ...\n");

    const int steps_per_thread = NUM_STEPS / NUM_THREADS;
    printf("Steps per thread: %.20f\n", steps_per_thread);

    double start = omp_get_wtime();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp single
        printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int start = tid * steps_per_thread;
        int end = start + steps_per_thread;
        double x;
        double t_sum = 0;

        printf("s,e=%d,%d\n", start,end);

        for (int i = start; i < end; i++) {
            // #pragma omp single
            // printf("for (int i = %d; %d < %d; i++)\n", start, i, end);

            x = (i+0.5) * step;
            t_sum += 4.0 / (1.0 + x*x);
        }

        #pragma omp atomic
        sum += t_sum;
    }

    pi = step * sum;

    double time = omp_get_wtime() - start;
    printf("Time calc: %f s\n", time);

    printf("π = %f\n", pi);

    return 0;
}

// # Load the numactl module to enable numa library linking
// module load numactl

// # Compile
// gcc -O3 -lm -lnuma --openmp sample.c -o sample

// # Run
// srun  sample valve.png valve-out.png
