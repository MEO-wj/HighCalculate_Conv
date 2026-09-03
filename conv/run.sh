#!/usr/bin/env bash

set -euo pipefail

# `run.sh` may be invoked by the evaluator as a non-login shell.  Initialise
# Environment Modules explicitly so `module load` is available in that mode,
# matching the robust setup used by the other赛道脚本.
if ! command -v module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -r /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# NUMA 5 is the most repeatable ordinary DDR node observed on the contest
# server. Keep auto-selection available for other hosts via NUMA_NODE=auto.
NUMA_NODE="${NUMA_NODE:-5}"
NUMA_CANDIDATES="${NUMA_CANDIDATES:-7,3,5,11,13,15,9,2,4,6,8,10,12,14}"
# Optional explicit memory NUMA node for controlled CPU-NUMA/HBM experiments.
# Empty (the default) preserves the competition's single-node `-N` policy.
MEMORY_NODE="${MEMORY_NODE:-}"
# Memory placement policy when MEMORY_NODE is set:
#   bind      force all anonymous allocations to MEMORY_NODE (current behavior)
#   preferred prefer MEMORY_NODE but allow fallback to the CPU NUMA node
MEMORY_POLICY="${MEMORY_POLICY:-bind}"
COMPILER="${COMPILER:-bisheng}"
BUILD_DIR="${BUILD_DIR:-.conv-build}"
TEST_RUNS="${TEST_RUNS:-5}"

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
    # 将三种预取策略编译成独立 worker，避免在 physical-row 热循环内分支。
    "${CC}" -O3 -ffast-math -funroll-loops -falign-loops=32 \
        -fno-plt -fomit-frame-pointer \
        -mcpu=native -msve-vector-bits=512 \
        -DCONV_SME_ASM_DOUBLE_BUFFER=1 \
        -DCONV_SVE_VECTORS=6 -DCONV_SME_INPUT_PREFETCH_LOCALITY=2 \
        -Dconv2d=conv2d_keep -fopenmp \
        -c conv2d.c -o "${BUILD_DIR}/conv2d_keep.o"
    "${CC}" -O3 -ffast-math -funroll-loops -falign-loops=32 \
        -fno-plt -fomit-frame-pointer \
        -mcpu=native -msve-vector-bits=512 \
        -DCONV_SME_ASM_DOUBLE_BUFFER=1 \
        -DCONV_SVE_VECTORS=6 -DCONV_SME_INPUT_PREFETCH_LOCALITY=0 \
        -Dconv2d=conv2d_stream -fopenmp \
        -c conv2d.c -o "${BUILD_DIR}/conv2d_stream.o"
    # 宽 kernel 使用更远的输入预取距离；这是通用宽度类别，不针对公开尺寸。
    "${CC}" -O3 -ffast-math -funroll-loops -falign-loops=32 \
        -fno-plt -fomit-frame-pointer \
        -mcpu=native -msve-vector-bits=512 \
        -DCONV_SME_ASM_DOUBLE_BUFFER=1 \
        -DCONV_SVE_VECTORS=6 -DCONV_SME_PREFETCH_MAX_WIDTH=128 \
        -DCONV_SME_INPUT_PREFETCH_DISTANCE=8 \
        -Dconv2d=conv2d_wide -fopenmp \
        -c conv2d.c -o "${BUILD_DIR}/conv2d_wide.o"
    "${CC}" -O3 -fopenmp -c conv2d_dispatch.c \
        -o "${BUILD_DIR}/conv2d_dispatch.o"
    "${CC}" -O3 -march=armv9-a+sme -msve-vector-bits=512 -c conv_sme_block.S \
        -o "${BUILD_DIR}/conv_sme_block.o"

    # 当前毕昇 SME 运行时需要 GCC 的异常展开支持，显式链接官方工具链归档。
    LIBGCC_EH=/home/HPC/HPCKit/25.2.0/compiler/gcc/lib64/gcc/aarch64-linux-gnu/12.3.1/libgcc_eh.a
    "${CC}" "${BUILD_DIR}/bench_conv.o" \
        "${BUILD_DIR}/conv2d_dispatch.o" \
        "${BUILD_DIR}/conv2d_keep.o" "${BUILD_DIR}/conv2d_stream.o" \
        "${BUILD_DIR}/conv2d_wide.o" \
        "${BUILD_DIR}/conv_sme_block.o" \
        "${LIBGCC_EH}" -o conv2d_test -lm -fopenmp
else
    # GCC 12 不提供本实现所用的 SME ACLE，保留 SVE 正确性/兼容性路径。
    "${CC}" "${COMMON_FLAGS[@]}" -ffast-math -funroll-loops \
        -DCONV_USE_SME=0 -DCONV_SVE_VECTORS=6 \
        -c conv2d.c -o "${BUILD_DIR}/conv2d.o"
    "${CC}" "${BUILD_DIR}/bench_conv.o" "${BUILD_DIR}/conv2d.o" \
        -o conv2d_test -lm -fopenmp
fi

select_idle_numa_node() {
    python3 - "${NUMA_CANDIDATES}" <<'PY'
import pathlib
import sys
import time


def parse_cpu_list(text):
    cpus = []
    for item in text.strip().split(","):
        if not item:
            continue
        if "-" in item:
            first, last = map(int, item.split("-", 1))
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(item))
    return cpus


def read_cpu_times():
    result = {}
    with open("/proc/stat", "r", encoding="ascii") as stat_file:
        for line in stat_file:
            fields = line.split()
            if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
                continue
            result[int(fields[0][3:])] = tuple(map(int, fields[1:]))
    return result


candidates = []
for value in sys.argv[1].split(","):
    node = int(value)
    cpulist = pathlib.Path(
        f"/sys/devices/system/node/node{node}/cpulist"
    )
    if cpulist.is_file():
        cpus = parse_cpu_list(cpulist.read_text(encoding="ascii"))
        if cpus:
            candidates.append((node, cpus))

if not candidates:
    raise SystemExit("No usable NUMA candidate")

before = read_cpu_times()
time.sleep(2.0)
after = read_cpu_times()

best = None
for order, (node, cpus) in enumerate(candidates):
    total = 0
    idle = 0
    for cpu in cpus:
        old = before[cpu]
        new = after[cpu]
        total += sum(new) - sum(old)
        idle += (new[3] - old[3]) + (new[4] - old[4])
    busy = 1.0 if total <= 0 else (total - idle) / total
    candidate = (busy, order, node)
    if best is None or candidate < best:
        best = candidate

print(best[2])
PY
}

if [[ "${NUMA_NODE}" == "auto" ]]; then
    NUMA_NODE="$(select_idle_numa_node)"
fi

if [[ ! "${NUMA_NODE}" =~ ^[0-9]+$ ]] || \
   [[ ! -r "/sys/devices/system/node/node${NUMA_NODE}/cpulist" ]]; then
    echo "Invalid NUMA_NODE=${NUMA_NODE}" >&2
    exit 2
fi

if [[ -n "${MEMORY_NODE}" ]] && { [[ ! "${MEMORY_NODE}" =~ ^[0-9]+$ ]] ||
    [[ ! -r "/sys/devices/system/node/node${MEMORY_NODE}/meminfo" ]]; }; then
    echo "Invalid MEMORY_NODE=${MEMORY_NODE}" >&2
    exit 2
fi

if [[ "${MEMORY_POLICY}" != "bind" && "${MEMORY_POLICY}" != "preferred" ]]; then
    echo "Invalid MEMORY_POLICY=${MEMORY_POLICY}; use bind or preferred." >&2
    exit 2
fi

if [[ "${MEMORY_POLICY}" == "preferred" && -z "${MEMORY_NODE}" ]]; then
    echo "MEMORY_POLICY=preferred requires MEMORY_NODE to be set." >&2
    exit 2
fi

if [[ ! "${TEST_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid TEST_RUNS=${TEST_RUNS}; use a positive integer." >&2
    exit 2
fi

echo "Using NUMA node ${NUMA_NODE} and ${TEST_RUNS} timed runs per CONV case." >&2

NUMACTL_ARGS=(-N "${NUMA_NODE}")
if [[ -n "${MEMORY_NODE}" ]]; then
    echo "Using memory NUMA node ${MEMORY_NODE} with policy ${MEMORY_POLICY}; verify contest single-NUMA compliance before submission." >&2
    if [[ "${MEMORY_POLICY}" == "preferred" ]]; then
        NUMACTL_ARGS+=(--preferred "${MEMORY_NODE}")
    else
        NUMACTL_ARGS+=(-m "${MEMORY_NODE}")
    fi
fi

numactl "${NUMACTL_ARGS[@]}" ./conv2d_test 4096 6144 39 39 "${TEST_RUNS}"
numactl "${NUMACTL_ARGS[@]}" ./conv2d_test 6144 4096 41 41 "${TEST_RUNS}"
numactl "${NUMACTL_ARGS[@]}" ./conv2d_test 4256 6390 55 55 "${TEST_RUNS}"
numactl "${NUMACTL_ARGS[@]}" ./conv2d_test 6390 4256 81 81 "${TEST_RUNS}"
