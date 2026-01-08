#ifndef COMMON_CPU_H
#define COMMON_CPU_H

#include <stdint.h>
#include <stdbool.h>

#ifndef TIDE_DTYPE
#define TIDE_DTYPE float
#endif

#ifndef TIDE_STENCIL
#define TIDE_STENCIL 4
#endif

#if defined(_OPENMP)
#if defined(_MSC_VER)
#define TIDE_OMP_INDEX int
#define TIDE_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE3 _Pragma("omp parallel for")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp parallel for")
#else
#define TIDE_OMP_INDEX int64_t
#define TIDE_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2)")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE3 _Pragma("omp parallel for collapse(3)")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp parallel for collapse(4)")
#endif
#else
#define TIDE_OMP_INDEX int64_t
#define TIDE_OMP_PARALLEL_FOR
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE3
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4
#endif

#endif
