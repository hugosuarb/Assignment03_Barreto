#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <time.h>
#include <sys/resource.h>
#endif

static inline uint64_t next_u64(uint64_t *state) {
    *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return *state;
}

static inline double u64_to_double(uint64_t x) {
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

void generate_matrix(double *M, int n, uint64_t seed) {
    uint64_t state = seed;
    for (int i = 0; i < n * n; i++) {
        M[i] = u64_to_double(next_u64(&state));
    }
}

void identity_matrix(double *M, int n) {
    memset(M, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        M[i * n + i] = 1.0;
    }
}

void clear_matrix(double *M, int n) {
    memset(M, 0, (size_t)n * n * sizeof(double));
}

double get_time_s(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

double get_memory_kb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.PeakWorkingSetSize / 1024.0;
    }
    return 0.0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (double)usage.ru_maxrss;
#endif
}

void multiply_seq(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        const double *a_row = &A[i * n];
        double *c_row = &C[i * n];
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a_row[k] * B[k * n + j];
            }
            c_row[j] = sum;
        }
    }
}

void multiply_omp(const double *A, const double *B, double *C, int n, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < n; i++) {
        const double *a_row = &A[i * n];
        double *c_row = &C[i * n];
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a_row[k] * B[k * n + j];
            }
            c_row[j] = sum;
        }
    }
}

double compute_checksum(const double *C, int n) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++) {
        sum += C[i];
    }
    return sum;
}

void run_kernel(const char *method, const double *A, const double *B, double *C, int n, int threads) {
    if (strcmp(method, "seq") == 0) {
        multiply_seq(A, B, C, n);
    } else {
        multiply_omp(A, B, C, n, threads);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        return 1;
    }
    const char *mode = argv[1];
    const char *method = argv[2];
    int n = atoi(argv[3]);
    int threads = atoi(argv[4]);
    int warmup = atoi(argv[5]);
    int reps = atoi(argv[6]);
    uint64_t seed = strtoull(argv[7], NULL, 10);

    if (strcmp(method, "seq") == 0) {
        threads = 1;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    size_t bytes = (size_t)n * n * sizeof(double);
    double *A = (double *)calloc((size_t)n * n, sizeof(double));
    double *B = (double *)calloc((size_t)n * n, sizeof(double));
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return 2;
    }

    if (strcmp(mode, "identity") == 0) {
        generate_matrix(A, n, seed);
        identity_matrix(B, n);
        clear_matrix(C, n);
        run_kernel(method, A, B, C, n, threads);
        double max_diff = 0.0;
        for (int i = 0; i < n * n; i++) {
            double d = fabs(C[i] - A[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("IDENTITY,%s,%d,%d,%.17e,%.17e\n", method, n, threads, max_diff, compute_checksum(C, n));
        free(A);
        free(B);
        free(C);
        return 0;
    }

    if (strcmp(mode, "zero") == 0) {
        generate_matrix(A, n, seed);
        clear_matrix(B, n);
        clear_matrix(C, n);
        run_kernel(method, A, B, C, n, threads);
        double max_diff = 0.0;
        for (int i = 0; i < n * n; i++) {
            double d = fabs(C[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("ZERO,%s,%d,%d,%.17e,%.17e\n", method, n, threads, max_diff, compute_checksum(C, n));
        free(A);
        free(B);
        free(C);
        return 0;
    }

    if (strcmp(mode, "dump") == 0) {
        generate_matrix(A, n, seed);
        generate_matrix(B, n, seed + 1);
        clear_matrix(C, n);
        run_kernel(method, A, B, C, n, threads);
        for (int i = 0; i < n * n; i++) {
            printf("%.17e%s", C[i], (i + 1 == n * n) ? "\n" : ",");
        }
        free(A);
        free(B);
        free(C);
        return 0;
    }

    generate_matrix(A, n, seed);
    generate_matrix(B, n, seed + 1);

    for (int w = 0; w < warmup; w++) {
        clear_matrix(C, n);
        run_kernel(method, A, B, C, n, threads);
    }

    const char *label = (strcmp(method, "seq") == 0) ? "Sequential_Baseline" : "OpenMP_Parallel";

    for (int r = 1; r <= reps; r++) {
        clear_matrix(C, n);
        double t0 = get_time_s();
        run_kernel(method, A, B, C, n, threads);
        double t1 = get_time_s();
        double elapsed = t1 - t0;
        double chk = compute_checksum(C, n);
        double mem_kb = get_memory_kb();
        printf("%s,%d,float64,%d,%d,%.9f,%.2f,%.12e\n", label, n, threads, r, elapsed, mem_kb, chk);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}