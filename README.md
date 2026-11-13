# Assignment 03: Parallel Matrix Multiplication with OpenMP

**Author:** Hugo Barreto Suárez  
**Course:** Big Data (3º Grado en Ciencia e Ingeniería de Datos - ULPGC)  
**Academic Year:** 2026–2027  
**Report:** [Assignment03_BarretoSuarez_Report.pdf](./Assignment03_BarretoSuarez_Report.pdf)

This repository contains the source code, benchmark scripts, raw measurements, generated figures, and the final research report for **Assignment 3**, which investigates how thread count ($p$) affects execution time, memory usage, speedup $S(p)$, and parallel efficiency $E(p)$ in a shared-memory OpenMP matrix multiplication implementation compared against its equivalent sequential baseline in C.

## 1. Repository Structure

    .
    ├── src/
    │   ├── baseline_matmul.c                 # Reused validated C implementation from Assignment 1
    │   └── matmul_a3.c                       # Equivalent sequential baseline + OpenMP parallel implementation
    ├── notebooks/
    │   └── experiments_and_analysis.ipynb    # Automated compilation, validation, benchmark & plotting
    ├── data/
    │   ├── raw_measurements.csv              # Raw measurements across all sizes, thread counts, and repetitions
    │   ├── summary_results.csv               # Aggregated medians, IQRs, Speedup S(p), and Efficiency E(p)
    │   └── execution_limits.csv              # Execution status per configuration
    ├── figures/
    │   ├── runtime_vs_threads.png            # Plot of median kernel execution time vs. thread count
    │   ├── speedup_vs_threads.png            # Plot of Speedup S(p) vs. thread count
    │   ├── efficiency_vs_threads.png         # Plot of Parallel Efficiency E(p) vs. thread count
    │   └── memory_vs_threads.png             # Plot of Peak Working Set memory vs. thread count
    ├── Assignment03_BarretoSuarez_Report.pdf # Final research report (PDF)
    └── README.md                             # Project documentation and instructions

## 2. Environment and Hardware Configuration

* **OS:** Windows 11 (64-bit)
* **CPU:** Intel Core Ultra 7 155H (16 physical cores / 22 logical processors)
* **RAM:** 15.37 GB
* **C Compiler & Flags:** GCC 16.2 with `-O2 -fopenmp -lpsapi`
* **Thread Affinity & Control:** `OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=true`
* **Python Environment:** CPython 3.14.6 (`numpy`, `pandas`, `matplotlib`, `psutil`, `jupyter`)

## 3. Reused Materials and Shared Input Generation

This study reuses the validated 1D contiguous row-major `double*` array layout and 64-bit Linear Congruential Generator (LCG) from Assignment 1 (`src/baseline_matmul.c`):

$$
\text{state} = (\text{state} \times 6364136223846793005 + 1442695040888963407) \pmod{2^{64}}
$$

Both `Sequential_Baseline` and `OpenMP_Parallel` execute the exact same $(i, j, k)$ loop order on identical IEEE 754 `float64` matrices $A$ (`seed = 12345`) and $B$ (`seed = 12346`). In the parallel version, rows $i$ of matrix $C$ are partitioned statically across threads (`#pragma omp parallel for num_threads(threads) schedule(static)`), ensuring race-free writes without synchronization locks.

## 4. Build and Run Instructions

### A. Compile C with OpenMP

    gcc -O2 -fopenmp src/matmul_a3.c -o src/matmul_a3.exe -lpsapi

### B. Command-Line Arguments

    <mode> <method> <n> <threads> <warmup_runs> <measured_reps> <seed>

Where `<mode>` is `identity`, `zero`, `dump`, or `benchmark`, and `<method>` is `seq` or `omp`.

Example manual runs for $n = 1000$, $2$ warm-ups, $5$ repetitions, and `seed = 12345`:

    ./src/matmul_a3.exe benchmark seq 1000 1 2 5 12345
    ./src/matmul_a3.exe benchmark omp 1000 1 2 5 12345
    ./src/matmul_a3.exe benchmark omp 1000 16 2 5 12345

## 5. Reproducing the Full Study and Plots

Open and execute all cells in `notebooks/experiments_and_analysis.ipynb`:

    jupyter notebook notebooks/experiments_and_analysis.ipynb