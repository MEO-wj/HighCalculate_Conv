#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

NUMA_NODE="${NUMA_NODE:-7}"
COMPILER="${COMPILER:-bisheng}"
BUILD_DIR="${BUILD_DIR:-.conv-build}"

module use /home/HPC/HPCKit/latest/modulefiles

case "${COMPILER}" in
    bisheng)
        module load bisheng/compiler5.0.0.2/bishengmodule
        CC=clang
        ;;
    gcc)
        module load gcc/compiler12.3.1/gccmodule
        CC=gcc
        ;;
    *)
        echo "Unsupported COMPILER=${COMPILER}; use bisheng or gcc." >&2
        exit 2
        ;;
esac

mkdir -p "${BUILD_DIR}"

# 保持官方测试程序的参考累加语义；fast-math 仅用于参赛内核。
COMMON_FLAGS=(-O3 -mcpu=native -msve-vector-bits=512 -fopenmp)
"${CC}" "${COMMON_FLAGS[@]}" -c bench_conv.c -o "${BUILD_DIR}/bench_conv.o"

if [[ "${COMPILER}" == "bisheng" ]]; then
    # SME ACLE 由毕昇 5.0 提供；四个 ZA tile 覆盖 16x64 输出块。
    "${CC}" -O3 -ffast-math -funroll-loops -falign-loops=32 \
        -fno-plt -fomit-frame-pointer \
        -mcpu=native -msve-vector-bits=512 \
        -DCONV_SVE_VECTORS=6 -fopenmp \
        -c conv2d.c -o "${BUILD_DIR}/conv2d.o"

    # 当前毕昇 SME 运行时需要 GCC 的异常展开支持，显式链接官方工具链归档。
    LIBGCC_EH=/home/HPC/HPCKit/25.2.0/compiler/gcc/lib64/gcc/aarch64-linux-gnu/12.3.1/libgcc_eh.a
    "${CC}" "${BUILD_DIR}/bench_conv.o" "${BUILD_DIR}/conv2d.o" \
        "${LIBGCC_EH}" -o conv2d_test -lm -fopenmp
else
    # GCC 12 不提供本实现所用的 SME ACLE，保留 SVE 正确性/兼容性路径。
    "${CC}" "${COMMON_FLAGS[@]}" -ffast-math -funroll-loops \
        -DCONV_USE_SME=0 -DCONV_SVE_VECTORS=6 \
        -c conv2d.c -o "${BUILD_DIR}/conv2d.o"
    "${CC}" "${BUILD_DIR}/bench_conv.o" "${BUILD_DIR}/conv2d.o" \
        -o conv2d_test -lm -fopenmp
fi

numactl -N "${NUMA_NODE}" ./conv2d_test 4096 6144 39 39 1
numactl -N "${NUMA_NODE}" ./conv2d_test 6144 4096 41 41 1
numactl -N "${NUMA_NODE}" ./conv2d_test 4256 6390 55 55 1
numactl -N "${NUMA_NODE}" ./conv2d_test 6390 4256 81 81 1
