#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#if defined(__linux__) && defined(__aarch64__) && defined(__ARM_FEATURE_SME)
#include <linux/prctl.h>
#include <sys/prctl.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

// 每个块同时计算多个相邻输出。向量 lane 对应不同的输出位置，
// 因而每个输出内部仍按照 kernel 行、列的原始顺序进行浮点累加。
#ifndef CONV_OUTPUT_BLOCK
#define CONV_OUTPUT_BLOCK 32
#endif

#ifndef CONV_SVE_VECTORS
#define CONV_SVE_VECTORS 6
#endif

#ifndef CONV_SVE_SLIDING
#define CONV_SVE_SLIDING 0
#endif

#ifndef CONV_SVE_VERTICAL
#define CONV_SVE_VERTICAL 1
#endif

#if CONV_SVE_VECTORS < 3 || CONV_SVE_VECTORS > 6
#error "CONV_SVE_VECTORS must be between 3 and 6"
#endif

#ifndef CONV_USE_SME
#define CONV_USE_SME 1
#endif

#ifndef CONV_SME_ITERATIVE_WINDOWS
#define CONV_SME_ITERATIVE_WINDOWS 1
#endif

#ifndef CONV_SME_CARRY_WINDOWS
#define CONV_SME_CARRY_WINDOWS 1
#endif

#ifndef CONV_SME_ASM_DOUBLE_BUFFER
#define CONV_SME_ASM_DOUBLE_BUFFER 0
#endif



/* Experimental 16x128 SME mapping; kept off for the normal submission path. */
#ifndef CONV_SME_WIDE128_EXPERIMENT
#define CONV_SME_WIDE128_EXPERIMENT 0
#endif

#ifndef CONV_SME_ALIGNED_PACKING
#define CONV_SME_ALIGNED_PACKING 1
#endif

/*
 * Reuse only the packed representation of unchanged convolution weights.
 * The cache is bounded to one entry per calling thread and validates the full
 * kernel contents, so changing data at the same address still rebuilds it.
 */
#ifndef CONV_SME_CACHE_PACKED_KERNEL
#define CONV_SME_CACHE_PACKED_KERNEL 1
#endif

#ifndef CONV_SME_PACKED_KERNEL_COPIES
#define CONV_SME_PACKED_KERNEL_COPIES 19
#endif

/*
 * Large packed-weight working sets put substantially more read pressure on
 * the shared cache hierarchy.  Give each pair of OpenMP workers a private
 * copy on a 38-core node, while retaining the compact single-copy layout for
 * smaller kernels.  The decision depends on packed bytes, not benchmark
 * rows/columns or any complete public test shape.
 */
#ifndef CONV_SME_PACKED_KERNEL_COPY_MIN_BYTES
#define CONV_SME_PACKED_KERNEL_COPY_MIN_BYTES (256U * 1024U)
#endif

#ifndef CONV_SME_PACKED_KERNEL_COPY_PAD_BYTES
#define CONV_SME_PACKED_KERNEL_COPY_PAD_BYTES 0
#endif

/*
 * Compile-time tuning knobs.  They describe generic kernel-width classes and
 * cache layout rather than any complete public benchmark shape, so candidate
 * values can be searched without specializing on rows/columns combinations.
 */
#ifndef CONV_SME_TASKS_SMALL
#define CONV_SME_TASKS_SMALL 4
#endif

#ifndef CONV_SME_TASKS_LARGE
#define CONV_SME_TASKS_LARGE 5
#endif

#ifndef CONV_SME_TASK_SPLIT_WIDTH
#define CONV_SME_TASK_SPLIT_WIDTH 64
#endif

#ifndef CONV_SME_PREFETCH_MIN_WIDTH
#define CONV_SME_PREFETCH_MIN_WIDTH 41
#endif

#ifndef CONV_SME_PREFETCH_MAX_WIDTH
#define CONV_SME_PREFETCH_MAX_WIDTH 64
#endif

#ifndef CONV_SME_PREFETCH_ROWS_MEDIUM
#define CONV_SME_PREFETCH_ROWS_MEDIUM 2
#endif

#ifndef CONV_SME_PREFETCH_ROWS_SMALL
#define CONV_SME_PREFETCH_ROWS_SMALL 1
#endif

#ifndef CONV_SME_PREFETCH_ROWS_LARGE
#define CONV_SME_PREFETCH_ROWS_LARGE 1
#endif

#ifndef CONV_SME_INPUT_PREFETCH_DISTANCE
#define CONV_SME_INPUT_PREFETCH_DISTANCE 4
#endif

#ifndef CONV_SME_INPUT_PREFETCH_LOCALITY
#define CONV_SME_INPUT_PREFETCH_LOCALITY 2
#endif

#ifndef CONV_SME_INTERLEAVE_WINDOWS
#define CONV_SME_INTERLEAVE_WINDOWS 0
#endif

#ifndef CONV_SME_TAIL_UNROLL
#define CONV_SME_TAIL_UNROLL 1
#endif

/*
 * Reuse the five live input windows for small/medium kernel tails.  This keeps
 * the original accumulation order but replaces four overlapping vector loads
 * per tail column with four register EXT operations and one predicated lane
 * refill.  Wider kernels retain the direct-load tail, which is more robust
 * under their higher cache and register pressure.
 */
#ifndef CONV_SME_CARRY_TAIL
#define CONV_SME_CARRY_TAIL 1
#endif

#ifndef CONV_SME_CARRY_TAIL_MAX_WIDTH
#define CONV_SME_CARRY_TAIL_MAX_WIDTH 64
#endif

#ifndef CONV_SME_GUIDED_MIN_TASKS_PER_THREAD
#define CONV_SME_GUIDED_MIN_TASKS_PER_THREAD 16
#endif

#ifndef CONV_SME_GUIDED_CHUNK
#define CONV_SME_GUIDED_CHUNK 1
#endif

#ifndef CONV_SME_BLOCKS_PER_TASK
#define CONV_SME_BLOCKS_PER_TASK 1
#endif

/*
 * Column-major task order.  Default row-major order makes one thread sweep a
 * full 64-column row of output tiles before advancing 16 rows, so the KH-1
 * overlapping input rows (a full-width working set) fall out of L2 and get
 * re-read from DRAM ~(KH+15)/16 times.  Column-major instead walks one narrow
 * column strip down through all row tiles, keeping the KH-1 row overlap (only
 * a few hundred bytes wide) resident in L1 and collapsing the vertical input
 * re-read to ~1x.  The accumulation order per output is unchanged.
 */
#ifndef CONV_SME_COLUMN_MAJOR
#define CONV_SME_COLUMN_MAJOR 0
#endif

/*
 * Fine static tasks remove the coarse column-partition tail on sufficiently
 * large problems.  Small problems retain the lower-overhead partition path;
 * wide kernels retain the separately validated guided schedule.
 */
#ifndef CONV_SME_FINE_TASK_MIN_PER_THREAD
#define CONV_SME_FINE_TASK_MIN_PER_THREAD 16
#endif

#if CONV_SME_BLOCKS_PER_TASK < 0
#error "CONV_SME_BLOCKS_PER_TASK must be non-negative"
#endif

#if CONV_SME_FINE_TASK_MIN_PER_THREAD < 1
#error "CONV_SME_FINE_TASK_MIN_PER_THREAD must be positive"
#endif

#if CONV_SME_TASKS_SMALL < 1 || CONV_SME_TASKS_LARGE < 1
#error "CONV SME task counts must be positive"
#endif

#if CONV_SME_GUIDED_MIN_TASKS_PER_THREAD < 1
#error "CONV SME guided scheduling threshold must be positive"
#endif

#if CONV_SME_GUIDED_CHUNK < 1
#error "CONV SME guided chunk must be positive"
#endif

#if CONV_SME_INTERLEAVE_WINDOWS != 0 && CONV_SME_INTERLEAVE_WINDOWS != 1
#error "CONV SME window interleave must be 0 or 1"
#endif

#if CONV_SME_PREFETCH_ROWS_SMALL < 0 || CONV_SME_PREFETCH_ROWS_MEDIUM < 0 || \
    CONV_SME_PREFETCH_ROWS_LARGE < 0
#error "CONV SME prefetch row counts must be non-negative"
#endif

#if CONV_SME_TAIL_UNROLL != 1 && CONV_SME_TAIL_UNROLL != 2 && \
    CONV_SME_TAIL_UNROLL != 4 && CONV_SME_TAIL_UNROLL != 8
#error "CONV_SME_TAIL_UNROLL must be 1, 2, 4, or 8"
#endif

#if CONV_SME_CARRY_WINDOWS && !CONV_SME_ITERATIVE_WINDOWS
#error "CONV_SME_CARRY_WINDOWS requires CONV_SME_ITERATIVE_WINDOWS"
#endif

#if CONV_SME_CACHE_PACKED_KERNEL != 0 && CONV_SME_CACHE_PACKED_KERNEL != 1
#error "CONV_SME_CACHE_PACKED_KERNEL must be 0 or 1"
#endif

#if CONV_SME_PACKED_KERNEL_COPIES < 1 || CONV_SME_PACKED_KERNEL_COPIES > 256
#error "CONV_SME_PACKED_KERNEL_COPIES must be between 1 and 256"
#endif

#if CONV_SME_PACKED_KERNEL_COPY_MIN_BYTES < 1
#error "CONV_SME_PACKED_KERNEL_COPY_MIN_BYTES must be positive"
#endif

#if CONV_SME_PACKED_KERNEL_COPY_PAD_BYTES < 0 || \
    (CONV_SME_PACKED_KERNEL_COPY_PAD_BYTES % 4) != 0
#error "CONV SME packed copy padding must be a non-negative float multiple"
#endif

#if defined(__linux__) && defined(__aarch64__) && defined(__ARM_FEATURE_SME) && \
    defined(PR_SME_SET_VL) && defined(PR_SME_GET_VL) &&                  \
    defined(PR_SME_VL_INHERIT) && defined(PR_SME_VL_LEN_MASK)
static int conv_sme_vl_64_ready;

/*
 * The fixed-width SME kernel uses sixteen FP32 lanes.  Configure the process
 * before main creates the OpenMP pool so every worker inherits the required
 * 64-byte streaming vector length.  This changes only the process ABI state,
 * not the machine-wide SME sysctl.
 */
__attribute__((constructor)) static void conv_configure_sme_vl(void)
{
    const unsigned long requested = 64UL | PR_SME_VL_INHERIT;
    if (prctl(PR_SME_SET_VL, requested, 0UL, 0UL, 0UL) < 0) {
        return;
    }

    const int actual = prctl(PR_SME_GET_VL, 0UL, 0UL, 0UL, 0UL);
    conv_sme_vl_64_ready =
        actual >= 0 && (actual & PR_SME_VL_LEN_MASK) == 64;
}
#else
static const int conv_sme_vl_64_ready = 0;
#endif

#if defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SVE_BITS) && \
    __ARM_FEATURE_SVE_BITS == 512 && CONV_USE_SME
#if CONV_SME_CACHE_PACKED_KERNEL
typedef struct {
    CONVFLOAT* packedKernel;
    CONVFLOAT* kernelSnapshot;
    size_t packedBytes;
    size_t kernelBytes;
    CONVINT kernelHeight;
    CONVINT kernelWidth;
    int packedCopies;
} ConvSmePackedKernelCache;

static _Thread_local ConvSmePackedKernelCache conv_sme_packed_cache;
#endif

/*
 * SME 主计算内核。
 *
 * 一个任务块最多计算 16 个输出行和 64 个输出列：
 *   - ZA 的 16 行对应 16 个相邻输出行；
 *   - ZA0、ZA1、ZA2、ZA3 分别对应四组 16 列输出；
 *   - packedKernel 提供 16 个输出行所需的 kernel 系数向量；
 *   - input 向量提供 16 个相邻输出列的数据；
 *   - FMOPA 将两者做外积并直接累加到 ZA。
 *
 * 函数在外层 OpenMP parallel 区域中由所有线程共同调用，内部 omp for
 * 只负责分配互不重叠的输出块，因此不需要锁或原子操作。
 */
__arm_new("za") __arm_locally_streaming static void
conv2d_sme_worker(const CONVFLOAT* __restrict__ input, CONVINT inputWidth,
                  const CONVFLOAT* __restrict__ packedKernelBase,
                  size_t packedCopyElements, int packedCopies,
                  CONVINT kernelHeight, CONVINT kernelWidth,
                  CONVFLOAT* __restrict__ output, CONVINT outputHeight,
                  CONVINT outputWidth, int useFineTasks)
{
    const uint32_t tileSize = 16;
    const size_t packedColumnStride = tileSize;
    const CONVFLOAT* const packedKernel =
        packedKernelBase +
        (packedCopies > 1
             ? (size_t)(((int64_t)omp_get_thread_num() * packedCopies) /
                        omp_get_num_threads()) *
                   packedCopyElements
             : 0);
    /*
     * 中小 kernel 在任务总数足够时使用细粒度 64 列任务；宽 kernel 将相邻
     * 列块保留在同一任务内。前者减少线程收尾差，后者保留顺序访存和缓存
     * 局部性。这里只按通用 kernel 宽度和任务量分类，不识别公开 Case。
     */
    const CONVINT partitionTasksPerRowTile =
        kernelWidth > CONV_SME_TASK_SPLIT_WIDTH
            ? CONV_SME_TASKS_LARGE
            : CONV_SME_TASKS_SMALL;
    const CONVINT totalRowTiles = (outputHeight + (CONVINT)tileSize - 1) /
                                  (CONVINT)tileSize;
    const CONVINT fullColBlocks = outputWidth / (4 * (CONVINT)tileSize);
#if CONV_SME_BLOCKS_PER_TASK > 0
    const CONVINT hasColTail =
        fullColBlocks * 4 * (CONVINT)tileSize < outputWidth;
    const CONVINT fullColTasks =
        (fullColBlocks + CONV_SME_BLOCKS_PER_TASK - 1) /
        CONV_SME_BLOCKS_PER_TASK;
    const CONVINT fineTasksPerRowTile =
        fullColTasks + hasColTail > 0 ? fullColTasks + hasColTail : 1;
    const CONVINT tasksPerRowTile =
        useFineTasks ? fineTasksPerRowTile : partitionTasksPerRowTile;
#else
    const CONVINT tasksPerRowTile = partitionTasksPerRowTile;
#endif
    const CONVINT totalTasks = totalRowTiles * tasksPerRowTile;
    /* schedule(runtime) 由 conv2d 根据工作量统一设为 static 或 guided。 */
#pragma omp for schedule(runtime)
    for (CONVINT task = 0; task < totalTasks; ++task) {
#if CONV_SME_COLUMN_MAJOR
        /* 列优先：task 先按列分区、再按行 tile 递增，让同一线程沿窄列带
         * 垂直扫过相邻 row tile，KH-1 行重叠工作集常驻 L1，消除垂直重读。 */
        const CONVINT rowTileIndex = task % totalRowTiles;
        const CONVINT colPartition = task / totalRowTiles;
#else
        const CONVINT rowTileIndex = task / tasksPerRowTile;
        const CONVINT colPartition = task % tasksPerRowTile;
#endif
        const CONVINT outputRowStart = rowTileIndex * (CONVINT)tileSize;
        const CONVINT rowCount =
            outputHeight - outputRowStart < (CONVINT)tileSize
                ? outputHeight - outputRowStart
                : (CONVINT)tileSize;
        const svbool_t activeRows = svwhilelt_b32((uint64_t)0, (uint64_t)rowCount);
        const CONVINT prefetchRowsPerStep =
            kernelWidth < CONV_SME_PREFETCH_MIN_WIDTH
                ? CONV_SME_PREFETCH_ROWS_SMALL
                : (kernelWidth <= CONV_SME_PREFETCH_MAX_WIDTH
                       ? CONV_SME_PREFETCH_ROWS_MEDIUM
                       : CONV_SME_PREFETCH_ROWS_LARGE);
        const CONVINT prefetchSteps =
            prefetchRowsPerStep > 0
                ? (rowCount + prefetchRowsPerStep - 1) / prefetchRowsPerStep
                : 0;
        const CONVINT prefetchStart =
            kernelHeight + rowCount - 1 - prefetchSteps;

        /*
         * 将当前 row tile 的完整 64 列块映射到本任务。
         * fine 模式通常每任务处理一个 64 列块；coarse 模式处理一个连续区间。
         */
#if CONV_SME_BLOCKS_PER_TASK > 0
        const CONVINT firstColBlock = useFineTasks
            ? (colPartition < fullColTasks
                   ? colPartition * CONV_SME_BLOCKS_PER_TASK
                   : fullColBlocks)
            : (CONVINT)(((int64_t)fullColBlocks * colPartition) /
                        tasksPerRowTile);
        const CONVINT fineLastColBlock =
            firstColBlock < fullColBlocks &&
                    firstColBlock + CONV_SME_BLOCKS_PER_TASK < fullColBlocks
                ? firstColBlock + CONV_SME_BLOCKS_PER_TASK
                : fullColBlocks;
        const CONVINT lastColBlock = useFineTasks
            ? fineLastColBlock
            : (CONVINT)(((int64_t)fullColBlocks * (colPartition + 1)) /
                        tasksPerRowTile);
#else
        const CONVINT firstColBlock =
            (CONVINT)(((int64_t)fullColBlocks * colPartition) / tasksPerRowTile);
        const CONVINT lastColBlock =
            (CONVINT)(((int64_t)fullColBlocks * (colPartition + 1)) /
                      tasksPerRowTile);
#endif
        const CONVINT fullColEnd = lastColBlock * 4 * (CONVINT)tileSize;
        CONVINT outputColStart = firstColBlock * 4 * (CONVINT)tileSize;
        for (; outputColStart < fullColEnd;
             outputColStart += 4 * (CONVINT)tileSize) {
            const svbool_t allCols = svptrue_b32();
            /* 每个 16×64 输出块从零开始，在 ZA 中完成全部累加后只写回一次。 */
            svzero_za();

            /*
             * 同时计算 rowCount 个相邻输出行会覆盖 KH+rowCount-1 个物理输入行。
             * packedRow 已将该物理行对应的 16 个 kernel 系数排成一个向量，
             * 无效的顶部/底部 lane 为 0，因此热循环中不需要逐 lane 分支。
             */
            for (CONVINT physicalRow = 0;
                 physicalRow < kernelHeight + rowCount - 1; ++physicalRow) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(outputRowStart + physicalRow) * inputWidth +
                    outputColStart;
                const CONVFLOAT* const packedRow =
                    packedKernel + (size_t)physicalRow * kernelWidth * tileSize;
#if CONV_SME_INPUT_PREFETCH_DISTANCE > 0
                /* 只对已实测受益的中等 kernel 宽度预取后续 input 行。 */
                if (kernelWidth >= CONV_SME_PREFETCH_MIN_WIDTH &&
                    kernelWidth <= CONV_SME_PREFETCH_MAX_WIDTH &&
                    physicalRow + CONV_SME_INPUT_PREFETCH_DISTANCE <
                    kernelHeight + rowCount - 1) {
                    const CONVFLOAT* const inputPrefetch =
                        input +
                        (size_t)(outputRowStart + physicalRow +
                                 CONV_SME_INPUT_PREFETCH_DISTANCE) *
                            inputWidth +
                        outputColStart;
                    __builtin_prefetch(inputPrefetch, 0,
                                       CONV_SME_INPUT_PREFETCH_LOCALITY);
                    const CONVINT inputTail = inputWidth - outputColStart;
                    if (inputTail > 47) {
                        __builtin_prefetch(inputPrefetch + 32, 0,
                                           CONV_SME_INPUT_PREFETCH_LOCALITY);
                    }
                    if (inputTail > 79) {
                        __builtin_prefetch(inputPrefetch + 64, 0,
                                           CONV_SME_INPUT_PREFETCH_LOCALITY);
                    }
                    if (inputTail > 111) {
                        __builtin_prefetch(inputPrefetch + 96, 0,
                                           CONV_SME_INPUT_PREFETCH_LOCALITY);
                    }
                }
#endif
                const CONVINT prefetchStep = physicalRow - prefetchStart;
                /*
                 * 在计算接近结束时逐步取得输出缓存行的写所有权，避免 16×64
                 * 结果同时写回时集中承担 write-allocate 延迟。
                 */
                if (prefetchStep >= 0 && prefetchStep < prefetchSteps) {
                    for (CONVINT prefetchOffset = 0;
                         prefetchOffset < prefetchRowsPerStep; ++prefetchOffset) {
                        const CONVINT prefetchRow =
                            prefetchStep * prefetchRowsPerStep + prefetchOffset;
                        if (prefetchRow < rowCount) {
                            CONVFLOAT* const outputPrefetch =
                                output +
                                (size_t)(outputRowStart + prefetchRow) * outputWidth +
                                outputColStart;
                            __builtin_prefetch(outputPrefetch, 1, 3);
                            __builtin_prefetch(outputPrefetch + tileSize, 1, 3);
                            __builtin_prefetch(outputPrefetch + 2 * tileSize, 1, 3);
                            __builtin_prefetch(outputPrefetch + 3 * tileSize, 1, 3);
                        }
                    }
                }

                CONVINT ik = 0;
                const svbool_t extensionLanes =
                    svwhilelt_b32((uint64_t)0, (uint64_t)15);
#if CONV_SME_ITERATIVE_WINDOWS && CONV_SME_CARRY_WINDOWS
                /*
                 * 五个连续 SVE 向量覆盖四组 16 列输出及右侧扩展数据。
                 * 完整 16 列 kernel 组之间继续携带窗口，减少重复 input 加载。
                 */
                const svfloat32_t zeroWindow = svdup_f32(0.0f);
                svfloat32_t window0 = zeroWindow;
                svfloat32_t window1 = zeroWindow;
                svfloat32_t window2 = zeroWindow;
                svfloat32_t window3 = zeroWindow;
                svfloat32_t window4 = zeroWindow;
#endif
#if CONV_SME_ASM_DOUBLE_BUFFER
                int asmUsed = 0;
                const CONVINT asmGroups =
                    rowCount == (CONVINT)tileSize && kernelWidth >= 32
                        ? kernelWidth / 16 - 1
                        : 0;
#endif
                for (; ik + 15 < kernelWidth; ik += 16) {
#if CONV_SME_ASM_DOUBLE_BUFFER
                    if (ik == 0 && asmGroups > 0) {
                        const CONVINT asmCount = asmGroups;
                        asm volatile(
                            "mov x0, %[in]\n"
                            "mov x1, %[pk]\n"
                            "mov x2, %[groups]\n"
                            "bl conv_sme_asm_block\n"
                            :
                            : [in] "r"(inputRow), [pk] "r"(packedRow),
                              [groups] "r"(asmCount)
                            : "memory", "x0", "x1", "x2", "x8", "x9",
                              "x30", "p0", "z0", "z1", "z2", "z3", "z4",
                              "z5", "z6", "z7");
                        asmUsed = 1;
                        ik = asmGroups * 16 - 16;
                        continue;
                    }
#endif
                    const CONVFLOAT* const inputBase = inputRow + ik;
#if CONV_SME_ITERATIVE_WINDOWS && CONV_SME_CARRY_WINDOWS
                    const svbool_t finalInputLanes =
                        ik + 31 < kernelWidth ||
                                (CONV_SME_CARRY_TAIL &&
                                 kernelWidth <= CONV_SME_CARRY_TAIL_MAX_WIDTH &&
                                 ik + 16 < kernelWidth)
                            ? allCols
                            : extensionLanes;
#if CONV_SME_ASM_DOUBLE_BUFFER
                    if (ik == 0 || asmUsed) {
                        asmUsed = 0;
#else
                    if (ik == 0) {
#endif
                        window0 = svld1_f32(allCols, inputBase);
                        window1 = svld1_f32(allCols, inputBase + tileSize);
                        window2 = svld1_f32(allCols, inputBase + 2 * tileSize);
                        window3 = svld1_f32(allCols, inputBase + 3 * tileSize);
                        window4 =
                            svld1_f32(finalInputLanes, inputBase + 4 * tileSize);
                    } else {
                        window0 = svext_f32(window0, window1, 1);
                        window1 = svext_f32(window1, window2, 1);
                        window2 = svext_f32(window2, window3, 1);
                        window3 = svext_f32(window3, window4, 1);
                        window4 =
                            svld1_f32(finalInputLanes, inputBase + 4 * tileSize);
                    }
                    const svfloat32_t inputValues0 = window0;
                    const svfloat32_t inputValues1 = window1;
                    const svfloat32_t inputValues2 = window2;
                    const svfloat32_t inputValues3 = window3;
#else
                    const svfloat32_t inputValues0 =
                        svld1_f32(allCols, inputBase);
                    const svfloat32_t inputValues1 =
                        svld1_f32(allCols, inputBase + tileSize);
                    const svfloat32_t inputValues2 =
                        svld1_f32(allCols, inputBase + 2 * tileSize);
                    const svfloat32_t inputValues3 =
                        svld1_f32(allCols, inputBase + 3 * tileSize);
                    const svfloat32_t inputValues4 =
                        svld1_f32(extensionLanes, inputBase + 4 * tileSize);
#endif

                    const svfloat32_t kernelValues0 =
                        svld1_f32(activeRows,
                                  packedRow + (size_t)ik * packedColumnStride);
                    /*
                     * 同一个 kernel 系数向量复用四次，分别更新 ZA0-ZA3；
                     * 一轮对应 16 个输出行 × 64 个输出列。
                     */
                    svmopa_za32_f32_m(0, activeRows, allCols, kernelValues0,
                                      inputValues0);
                    svmopa_za32_f32_m(1, activeRows, allCols, kernelValues0,
                                      inputValues1);
                    svmopa_za32_f32_m(2, activeRows, allCols, kernelValues0,
                                      inputValues2);
                    svmopa_za32_f32_m(3, activeRows, allCols, kernelValues0,
                                      inputValues3);

#if CONV_SME_ITERATIVE_WINDOWS
#if !CONV_SME_CARRY_WINDOWS
                    svfloat32_t window0 = inputValues0;
                    svfloat32_t window1 = inputValues1;
                    svfloat32_t window2 = inputValues2;
                    svfloat32_t window3 = inputValues3;
                    svfloat32_t window4 = inputValues4;
                    const svfloat32_t zeroWindow = svdup_f32(0.0f);
#endif

                    /*
                     * kernel 向右移动一列时，四个 input 窗口也右移一 lane。
                     * EXT 复用相邻向量重叠数据，避免重新执行四个完整 LD1W。
                     */
#if CONV_SME_INTERLEAVE_WINDOWS
#define CONV_SME_SLIDE_STEP(SHIFT)                                              \
                    do {                                                        \
                        const svfloat32_t kernelValues = svld1_f32(             \
                            activeRows, packedRow +                            \
                                            (size_t)(ik + (SHIFT)) *           \
                                                packedColumnStride);           \
                        window0 = svext_f32(window0, window1, 1);               \
                        window1 = svext_f32(window1, window2, 1);               \
                        svmopa_za32_f32_m(0, activeRows, allCols, kernelValues, \
                                          window0);                             \
                        svmopa_za32_f32_m(1, activeRows, allCols, kernelValues, \
                                          window1);                             \
                        window2 = svext_f32(window2, window3, 1);               \
                        window3 = svext_f32(window3, window4, 1);               \
                        svmopa_za32_f32_m(2, activeRows, allCols, kernelValues, \
                                          window2);                             \
                        svmopa_za32_f32_m(3, activeRows, allCols, kernelValues, \
                                          window3);                             \
                        window4 = svext_f32(window4, zeroWindow, 1);            \
                    } while (0)
#else
#define CONV_SME_SLIDE_STEP(SHIFT)                                              \
                    do {                                                        \
                        const svfloat32_t kernelValues = svld1_f32(             \
                            activeRows, packedRow +                            \
                                            (size_t)(ik + (SHIFT)) *           \
                                                packedColumnStride);           \
                        window0 = svext_f32(window0, window1, 1);               \
                        window1 = svext_f32(window1, window2, 1);               \
                        window2 = svext_f32(window2, window3, 1);               \
                        window3 = svext_f32(window3, window4, 1);               \
                        window4 = svext_f32(window4, zeroWindow, 1);            \
                        svmopa_za32_f32_m(0, activeRows, allCols, kernelValues, \
                                          window0);                             \
                        svmopa_za32_f32_m(1, activeRows, allCols, kernelValues, \
                                          window1);                             \
                        svmopa_za32_f32_m(2, activeRows, allCols, kernelValues, \
                                          window2);                             \
                        svmopa_za32_f32_m(3, activeRows, allCols, kernelValues, \
                                          window3);                             \
                    } while (0)
#endif

#if !CONV_SME_CARRY_WINDOWS
#define CONV_SME_SLIDE_FINAL(SHIFT)                                             \
                    do {                                                        \
                        const svfloat32_t kernelValues = svld1_f32(             \
                            activeRows, packedRow +                            \
                                            (size_t)(ik + (SHIFT)) *           \
                                                packedColumnStride);           \
                        window0 = svext_f32(window0, window1, 1);               \
                        window1 = svext_f32(window1, window2, 1);               \
                        window2 = svext_f32(window2, window3, 1);               \
                        window3 = svext_f32(window3, window4, 1);               \
                        svmopa_za32_f32_m(0, activeRows, allCols, kernelValues, \
                                          window0);                             \
                        svmopa_za32_f32_m(1, activeRows, allCols, kernelValues, \
                                          window1);                             \
                        svmopa_za32_f32_m(2, activeRows, allCols, kernelValues, \
                                          window2);                             \
                        svmopa_za32_f32_m(3, activeRows, allCols, kernelValues, \
                                          window3);                             \
                    } while (0)
#endif
#else
#define CONV_SME_SLIDE_STEP(SHIFT)                                                 \
                    do {                                                           \
                        const svfloat32_t kernelValues = svld1_f32(                 \
                            activeRows, packedRow +                                \
                                            (size_t)(ik + (SHIFT)) *               \
                                                packedColumnStride);               \
                        svmopa_za32_f32_m(                                          \
                            0, activeRows, allCols, kernelValues,                  \
                            svext_f32(inputValues0, inputValues1, (SHIFT)));         \
                        svmopa_za32_f32_m(                                          \
                            1, activeRows, allCols, kernelValues,                  \
                            svext_f32(inputValues1, inputValues2, (SHIFT)));         \
                        svmopa_za32_f32_m(                                          \
                            2, activeRows, allCols, kernelValues,                  \
                            svext_f32(inputValues2, inputValues3, (SHIFT)));         \
                        svmopa_za32_f32_m(                                          \
                            3, activeRows, allCols, kernelValues,                  \
                            svext_f32(inputValues3, inputValues4, (SHIFT)));         \
                    } while (0)
#endif

                    CONV_SME_SLIDE_STEP(1);
                    CONV_SME_SLIDE_STEP(2);
                    CONV_SME_SLIDE_STEP(3);
                    CONV_SME_SLIDE_STEP(4);
                    CONV_SME_SLIDE_STEP(5);
                    CONV_SME_SLIDE_STEP(6);
                    CONV_SME_SLIDE_STEP(7);
                    CONV_SME_SLIDE_STEP(8);
                    CONV_SME_SLIDE_STEP(9);
                    CONV_SME_SLIDE_STEP(10);
                    CONV_SME_SLIDE_STEP(11);
                    CONV_SME_SLIDE_STEP(12);
                    CONV_SME_SLIDE_STEP(13);
                    CONV_SME_SLIDE_STEP(14);
#if CONV_SME_ITERATIVE_WINDOWS && !CONV_SME_CARRY_WINDOWS
                    CONV_SME_SLIDE_FINAL(15);
#else
                    CONV_SME_SLIDE_STEP(15);
#endif
#undef CONV_SME_SLIDE_STEP
#if CONV_SME_ITERATIVE_WINDOWS && !CONV_SME_CARRY_WINDOWS
#undef CONV_SME_SLIDE_FINAL
#endif
                }

#if CONV_SME_CARRY_TAIL && CONV_SME_ITERATIVE_WINDOWS && \
    CONV_SME_CARRY_WINDOWS
                /*
                 * 中小 kernel 的不足 16 列尾部继续使用已有窗口；每步只加载
                 * 最右侧新进入的一个 lane。宽 kernel 经实测采用下方直接加载更稳。
                 */
                if (ik > 0 && ik < kernelWidth &&
                    kernelWidth <= CONV_SME_CARRY_TAIL_MAX_WIDTH) {
                    const svbool_t oneInputLane =
                        svwhilelt_b32((uint64_t)0, (uint64_t)1);
                    for (; ik < kernelWidth; ++ik) {
                        const svfloat32_t kernelValues =
                            svld1_f32(activeRows,
                                      packedRow +
                                          (size_t)ik * packedColumnStride);
                        window0 = svext_f32(window0, window1, 1);
                        window1 = svext_f32(window1, window2, 1);
                        window2 = svext_f32(window2, window3, 1);
                        window3 = svext_f32(window3, window4, 1);
                        svmopa_za32_f32_m(0, activeRows, allCols, kernelValues,
                                          window0);
                        svmopa_za32_f32_m(1, activeRows, allCols, kernelValues,
                                          window1);
                        svmopa_za32_f32_m(2, activeRows, allCols, kernelValues,
                                          window2);
                        svmopa_za32_f32_m(3, activeRows, allCols, kernelValues,
                                          window3);
                        if (ik + 1 < kernelWidth) {
                            window4 = svld1_f32(
                                oneInputLane,
                                inputRow + ik + 4 * (CONVINT)tileSize);
                        }
                    }
                }
#endif

#if CONV_SME_TAIL_UNROLL > 1
#pragma clang loop unroll_count(CONV_SME_TAIL_UNROLL)
#endif
                /*
                 * 通用 kernel 列尾：直接加载四个 input 向量并继续 FMOPA。
                 * 该路径也覆盖未启用 carry-tail 的宽 kernel，保持任意合法宽度正确。
                 */
                for (; ik < kernelWidth; ++ik) {
                    const svfloat32_t kernelValues =
                        svld1_f32(activeRows,
                                  packedRow + (size_t)ik * packedColumnStride);
                    const CONVFLOAT* const inputBase = inputRow + ik;
                    const svfloat32_t inputValues0 =
                        svld1_f32(allCols, inputBase);
                    const svfloat32_t inputValues1 =
                        svld1_f32(allCols, inputBase + tileSize);
                    const svfloat32_t inputValues2 =
                        svld1_f32(allCols, inputBase + 2 * tileSize);
                    const svfloat32_t inputValues3 =
                        svld1_f32(allCols, inputBase + 3 * tileSize);
                    svmopa_za32_f32_m(0, activeRows, allCols, kernelValues,
                                      inputValues0);
                    svmopa_za32_f32_m(1, activeRows, allCols, kernelValues,
                                      inputValues1);
                    svmopa_za32_f32_m(2, activeRows, allCols, kernelValues,
                                      inputValues2);
                    svmopa_za32_f32_m(3, activeRows, allCols, kernelValues,
                                      inputValues3);
                }
            }

            /* ZA 中已经是最终结果；逐行写回四个 16 列 tile，不保存中间部分和。 */
            for (CONVINT row = 0; row < rowCount; ++row) {
                CONVFLOAT* const outputBase =
                    output + (size_t)(outputRowStart + row) * outputWidth +
                    outputColStart;
                svst1_hor_za32(0, (uint32_t)row, allCols, outputBase);
                svst1_hor_za32(1, (uint32_t)row, allCols,
                               outputBase + tileSize);
                svst1_hor_za32(2, (uint32_t)row, allCols,
                               outputBase + 2 * tileSize);
                svst1_hor_za32(3, (uint32_t)row, allCols,
                               outputBase + 3 * tileSize);
            }
        }

        /* 输出宽度不足 64 列的最终部分只交给该 row tile 的最后一个任务。 */
        if (colPartition != tasksPerRowTile - 1) {
            continue;
        }

        for (outputColStart = fullColBlocks * 4 * (CONVINT)tileSize;
             outputColStart < outputWidth; outputColStart += (CONVINT)tileSize) {
            const CONVINT colCount =
                outputWidth - outputColStart < (CONVINT)tileSize
                    ? outputWidth - outputColStart
                    : (CONVINT)tileSize;
            const svbool_t activeCols =
                svwhilelt_b32((uint64_t)0, (uint64_t)colCount);
            /* 列尾改用谓词化单 ZA 16×16 内核，禁止越界读写。 */
            svzero_za();

            for (CONVINT physicalRow = 0;
                 physicalRow < kernelHeight + rowCount - 1; ++physicalRow) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(outputRowStart + physicalRow) * inputWidth +
                    outputColStart;
                const CONVFLOAT* const packedRow =
                    packedKernel + (size_t)physicalRow * kernelWidth * tileSize;

                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const svfloat32_t kernelValues =
                        svld1_f32(activeRows,
                                  packedRow + (size_t)ik * packedColumnStride);
                    const svfloat32_t inputValues =
                        svld1_f32(activeCols, inputRow + ik);
                    svmopa_za32_f32_m(0, activeRows, activeCols, kernelValues,
                                      inputValues);
                }
            }

            for (CONVINT row = 0; row < rowCount; ++row) {
                svst1_hor_za32(0, (uint32_t)row, activeCols,
                               output + (size_t)(outputRowStart + row) * outputWidth +
                                   outputColStart);
            }
        }
    }
}

#if CONV_SME_WIDE128_EXPERIMENT
/*
 * Isolated 16x128 experiment.  It deliberately keeps the original
 * physical-row/column accumulation order, but uses eight ZA tiles so one
 * packed kernel vector is reused across 128 output columns.
 */
__arm_new("za") __arm_locally_streaming static void
conv2d_sme_worker_wide128(const CONVFLOAT* __restrict__ input,
                          CONVINT inputWidth,
                          const CONVFLOAT* __restrict__ packedKernelBase,
                          size_t packedCopyElements, int packedCopies,
                          CONVINT kernelHeight, CONVINT kernelWidth,
                          CONVFLOAT* __restrict__ output,
                          CONVINT outputHeight, CONVINT outputWidth)
{
    const CONVINT tile = 16;
    const CONVFLOAT* const packedKernel =
        packedKernelBase +
        (packedCopies > 1
             ? (size_t)(((int64_t)omp_get_thread_num() * packedCopies) /
                        omp_get_num_threads()) * packedCopyElements
             : 0);
    const CONVINT rowTiles = (outputHeight + tile - 1) / tile;
    const CONVINT full128 = outputWidth / (8 * tile);
    const CONVINT totalTasks = rowTiles * (full128 + (outputWidth % (8 * tile) != 0));

#pragma omp for schedule(static)
    for (CONVINT task = 0; task < totalTasks; ++task) {
        const CONVINT parts = full128 + (outputWidth % (8 * tile) != 0);
        const CONVINT rowTile = task / parts;
        const CONVINT part = task % parts;
        const CONVINT rowStart = rowTile * tile;
        const CONVINT rowCount = outputHeight - rowStart < tile
                                     ? outputHeight - rowStart : tile;
        const svbool_t rows = svwhilelt_b32(0, (uint64_t)rowCount);
        const CONVINT colStart = part < full128 ? part * 8 * tile : full128 * 8 * tile;
        const CONVINT colEnd = part < full128 ? colStart + 8 * tile : outputWidth;

        for (CONVINT col = colStart; col < colEnd; col += 8 * tile) {
            const CONVINT colsThis = colEnd - col >= 8 * tile ? 8 * tile : colEnd - col;
            if (colsThis == 8 * tile) {
                const svbool_t all = svptrue_b32();
                svzero_za();
                for (CONVINT physical = 0;
                     physical < kernelHeight + rowCount - 1; ++physical) {
                    const CONVFLOAT* in = input + (size_t)(rowStart + physical) * inputWidth + col;
                    const CONVFLOAT* pk = packedKernel + (size_t)physical * kernelWidth * tile;
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                        const svfloat32_t kv = svld1_f32(rows, pk + (size_t)ik * tile);
                        const CONVFLOAT* base = in + ik;
                        svmopa_za32_f32_m(0, rows, all, kv, svld1_f32(all, base + 0 * tile));
                        svmopa_za32_f32_m(1, rows, all, kv, svld1_f32(all, base + 1 * tile));
                        svmopa_za32_f32_m(2, rows, all, kv, svld1_f32(all, base + 2 * tile));
                        svmopa_za32_f32_m(3, rows, all, kv, svld1_f32(all, base + 3 * tile));
                        svmopa_za32_f32_m(4, rows, all, kv, svld1_f32(all, base + 4 * tile));
                        svmopa_za32_f32_m(5, rows, all, kv, svld1_f32(all, base + 5 * tile));
                        svmopa_za32_f32_m(6, rows, all, kv, svld1_f32(all, base + 6 * tile));
                        svmopa_za32_f32_m(7, rows, all, kv, svld1_f32(all, base + 7 * tile));
                    }
                }
                for (CONVINT r = 0; r < rowCount; ++r) {
                    CONVFLOAT* out = output + (size_t)(rowStart + r) * outputWidth + col;
                    svst1_hor_za32(0, (uint32_t)r, all, out + 0 * tile);
                    svst1_hor_za32(1, (uint32_t)r, all, out + 1 * tile);
                    svst1_hor_za32(2, (uint32_t)r, all, out + 2 * tile);
                    svst1_hor_za32(3, (uint32_t)r, all, out + 3 * tile);
                    svst1_hor_za32(4, (uint32_t)r, all, out + 4 * tile);
                    svst1_hor_za32(5, (uint32_t)r, all, out + 5 * tile);
                    svst1_hor_za32(6, (uint32_t)r, all, out + 6 * tile);
                    svst1_hor_za32(7, (uint32_t)r, all, out + 7 * tile);
                }
            }
        }

        if (part == full128 && outputWidth % (8 * tile) != 0) {
            for (CONVINT col = colStart; col < outputWidth; col += tile) {
                const CONVINT n = outputWidth - col < tile ? outputWidth - col : tile;
                const svbool_t cols = svwhilelt_b32(0, (uint64_t)n);
                svzero_za();
                for (CONVINT physical = 0;
                     physical < kernelHeight + rowCount - 1; ++physical) {
                    const CONVFLOAT* in = input + (size_t)(rowStart + physical) * inputWidth + col;
                    const CONVFLOAT* pk = packedKernel + (size_t)physical * kernelWidth * tile;
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                        const svfloat32_t kv = svld1_f32(rows, pk + (size_t)ik * tile);
                        svmopa_za32_f32_m(0, rows, cols, kv,
                                          svld1_f32(cols, in + ik));
                    }
                }
                for (CONVINT r = 0; r < rowCount; ++r) {
                    svst1_hor_za32(0, (uint32_t)r, cols,
                                   output + (size_t)(rowStart + r) * outputWidth + col);
                }
            }
        }
    }
}
#endif

#endif


void conv2d(const CONVFLOAT* __restrict__ input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* __restrict__ kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* __restrict__ output)
{
    /* valid convolution：不填充，输出尺寸由输入尺寸减去 kernel 尺寸得到。 */
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#if defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SVE_BITS) && \
    __ARM_FEATURE_SVE_BITS == 512 && CONV_USE_SME
    /* 只有成功配置 512 位 streaming vector length 时才进入固定 16-lane SME 路径。 */
    if (conv_sme_vl_64_ready) {
    const size_t tileSize = 16;
    /*
     * 16 个相邻输出行共同需要 KH+15 个物理输入行。打包时为每个
     * (physicalRow, kernelColumn) 保存一个 16-lane kernel 系数向量。
     */
    const size_t packedPhysicalRows = (size_t)kernelHeight + tileSize - 1;
    size_t packedGroups = 0;

    if (kernelHeight > 0 && kernelWidth > 0 &&
        packedPhysicalRows <= SIZE_MAX / (size_t)kernelWidth) {
        packedGroups = packedPhysicalRows * (size_t)kernelWidth;
    }

    const size_t packedGroupElements = tileSize;
    if (packedGroups > 0 &&
        packedGroups <= SIZE_MAX / packedGroupElements &&
        packedGroups * packedGroupElements <= SIZE_MAX / sizeof(CONVFLOAT)) {
        const size_t packedElements = packedGroups * packedGroupElements;
        const size_t packedBytes = packedElements * sizeof(CONVFLOAT);
        const size_t packedCopyPadElements =
            (size_t)CONV_SME_PACKED_KERNEL_COPY_PAD_BYTES / sizeof(CONVFLOAT);
        const size_t packedCopyElements =
            packedElements <= SIZE_MAX - packedCopyPadElements
                ? packedElements + packedCopyPadElements
                : 0;
        const size_t kernelElements =
            (size_t)kernelHeight * (size_t)kernelWidth;
        const size_t kernelBytes = kernelElements * sizeof(CONVFLOAT);
        const int maximumPackedCopies = omp_get_max_threads();
        const int configuredPackedCopies =
            packedBytes >= (size_t)CONV_SME_PACKED_KERNEL_COPY_MIN_BYTES
                ? CONV_SME_PACKED_KERNEL_COPIES
                : 1;
        const int requestedPackedCopies =
            configuredPackedCopies < maximumPackedCopies
                ? configuredPackedCopies
                : maximumPackedCopies;
        const int packedCopies =
            requestedPackedCopies > 0 ? requestedPackedCopies : 1;
        const size_t packedTotalBytes =
            packedCopyElements > 0 &&
                    packedCopyElements <=
                        SIZE_MAX / sizeof(CONVFLOAT) / (size_t)packedCopies
                ? packedCopyElements * (size_t)packedCopies * sizeof(CONVFLOAT)
                : 0;
        int packedKernelNeedsBuild = 1;
        CONVFLOAT* kernelSnapshot = NULL;
#if CONV_SME_ALIGNED_PACKING
        /* 一个 packed 16-lane FP32 向量恰好为 64 字节，与缓存行对齐。 */
        CONVFLOAT* packedKernel = NULL;
#else
        CONVFLOAT* packedKernel = NULL;
#endif
#if CONV_SME_CACHE_PACKED_KERNEL
        if (conv_sme_packed_cache.packedKernel != NULL &&
            conv_sme_packed_cache.kernelSnapshot != NULL &&
            conv_sme_packed_cache.packedBytes == packedTotalBytes &&
            conv_sme_packed_cache.kernelBytes == kernelBytes &&
            conv_sme_packed_cache.kernelHeight == kernelHeight &&
            conv_sme_packed_cache.kernelWidth == kernelWidth &&
            conv_sme_packed_cache.packedCopies == packedCopies &&
            memcmp(conv_sme_packed_cache.kernelSnapshot, kernel, kernelBytes) ==
                0) {
            packedKernel = conv_sme_packed_cache.packedKernel;
            packedKernelNeedsBuild = 0;
        }
#endif
        if (packedKernel == NULL && packedTotalBytes > 0) {
#if CONV_SME_ALIGNED_PACKING
            packedKernel = (CONVFLOAT*)aligned_alloc(64, packedTotalBytes);
#else
            packedKernel = (CONVFLOAT*)malloc(packedTotalBytes);
#endif
#if CONV_SME_CACHE_PACKED_KERNEL
            if (packedKernel != NULL) {
                kernelSnapshot = (CONVFLOAT*)malloc(kernelBytes);
                if (kernelSnapshot != NULL) {
                    memcpy(kernelSnapshot, kernel, kernelBytes);
                }
            }
#endif
        }
        if (packedKernel != NULL) {
            /*
             * 调度选择只依赖 kernel 宽度、输出 tile 数和线程数：
             * 任务足够多的中小 kernel 使用 fine-static；宽 kernel 使用
             * coarse-guided；小问题回退 coarse-static，避免调度开销。
             */
            const CONVINT scheduleRowTiles =
                (outputHeight + (CONVINT)tileSize - 1) / (CONVINT)tileSize;
            const CONVINT partitionScheduleTasksPerRowTile =
                kernelWidth > CONV_SME_TASK_SPLIT_WIDTH
                    ? CONV_SME_TASKS_LARGE
                    : CONV_SME_TASKS_SMALL;
#if CONV_SME_BLOCKS_PER_TASK > 0
            const CONVINT scheduleFullColBlocks =
                outputWidth / (4 * (CONVINT)tileSize);
            const CONVINT fineScheduleTasksPerRowTile =
                (scheduleFullColBlocks + CONV_SME_BLOCKS_PER_TASK - 1) /
                    CONV_SME_BLOCKS_PER_TASK +
                ((outputWidth % (4 * (CONVINT)tileSize)) != 0);
#else
            const CONVINT fineScheduleTasksPerRowTile = 0;
#endif
            const int scheduleThreads = omp_get_max_threads();
            const int useFineTasks =
                kernelWidth <= CONV_SME_TASK_SPLIT_WIDTH &&
                fineScheduleTasksPerRowTile > 0 && scheduleThreads > 0 &&
                (int64_t)scheduleRowTiles * fineScheduleTasksPerRowTile >=
                    (int64_t)scheduleThreads *
                        CONV_SME_FINE_TASK_MIN_PER_THREAD;
            const CONVINT scheduleTasksPerRowTile =
                useFineTasks ? fineScheduleTasksPerRowTile
                             : partitionScheduleTasksPerRowTile;
            const int64_t scheduleTasks =
                (int64_t)scheduleRowTiles * scheduleTasksPerRowTile;
            /* 宽 kernel 任务更重，guided(1) 用于减少最后少数线程的长尾。 */
            const int useGuidedSchedule =
                !useFineTasks && kernelWidth > CONV_SME_TASK_SPLIT_WIDTH &&
                scheduleThreads > 0 &&
                scheduleTasks >=
                    (int64_t)scheduleThreads *
                        CONV_SME_GUIDED_MIN_TASKS_PER_THREAD;
            omp_sched_t previousSchedule;
            int previousChunkSize;
            omp_get_schedule(&previousSchedule, &previousChunkSize);
            omp_set_schedule(useGuidedSchedule ? omp_sched_guided
                                               : omp_sched_static,
                             useGuidedSchedule ? CONV_SME_GUIDED_CHUNK : 0);
#pragma omp parallel
            {
                /*
                 * 多线程并行打包 kernel。jk = physicalRow-outputRow 保持每个
                 * 输出元素原有的 jk -> ik 累加顺序；边界外系数填 0。
                 */
                if (packedKernelNeedsBuild) {
                    if (packedCopies > 1) {
#pragma omp for schedule(static)
                        for (int packedCopy = 0; packedCopy < packedCopies;
                             ++packedCopy) {
                            CONVFLOAT* const threadPackedKernel =
                                packedKernel +
                                (size_t)packedCopy * packedCopyElements;
                            for (size_t group = 0; group < packedGroups;
                                 ++group) {
                                const CONVINT physicalRow =
                                    (CONVINT)(group / (size_t)kernelWidth);
                                const CONVINT ik =
                                    (CONVINT)(group % (size_t)kernelWidth);
                                CONVFLOAT* const packedValues =
                                    threadPackedKernel + group * tileSize;
                                for (CONVINT outputRow = 0;
                                     outputRow < (CONVINT)tileSize;
                                     ++outputRow) {
                                    const CONVINT jk = physicalRow - outputRow;
                                    packedValues[outputRow] =
                                        (jk >= 0 && jk < kernelHeight)
                                            ? kernel[(size_t)jk * kernelWidth +
                                                     ik]
                                            : 0.0f;
                                }
                            }
                        }
                    } else {
#pragma omp for schedule(static)
                        for (size_t group = 0; group < packedGroups; ++group) {
                            const CONVINT physicalRow =
                                (CONVINT)(group / (size_t)kernelWidth);
                            const CONVINT ik =
                                (CONVINT)(group % (size_t)kernelWidth);
                            CONVFLOAT* const packedValues =
                                packedKernel + group * tileSize;
                            for (CONVINT outputRow = 0;
                                 outputRow < (CONVINT)tileSize; ++outputRow) {
                                const CONVINT jk = physicalRow - outputRow;
                                packedValues[outputRow] =
                                    (jk >= 0 && jk < kernelHeight)
                                        ? kernel[(size_t)jk * kernelWidth + ik]
                                        : 0.0f;
                            }
                        }
                    }
                }

#if CONV_SME_WIDE128_EXPERIMENT
                conv2d_sme_worker_wide128(
                    input, inputWidth, packedKernel, packedCopyElements,
                    packedCopies, kernelHeight, kernelWidth, output,
                    outputHeight, outputWidth);
#else
                conv2d_sme_worker(input, inputWidth, packedKernel,
                                  packedCopyElements, packedCopies, kernelHeight,
                                  kernelWidth, output, outputHeight, outputWidth,
                                  useFineTasks);
#endif
            }
            /* 恢复调用者原有 OpenMP runtime 调度状态，避免污染外部程序。 */
            omp_set_schedule(previousSchedule, previousChunkSize);
#if CONV_SME_CACHE_PACKED_KERNEL
            if (packedKernelNeedsBuild && kernelSnapshot != NULL) {
                free(conv_sme_packed_cache.packedKernel);
                free(conv_sme_packed_cache.kernelSnapshot);
                conv_sme_packed_cache.packedKernel = packedKernel;
                conv_sme_packed_cache.kernelSnapshot = kernelSnapshot;
                conv_sme_packed_cache.packedBytes = packedTotalBytes;
                conv_sme_packed_cache.kernelBytes = kernelBytes;
                conv_sme_packed_cache.kernelHeight = kernelHeight;
                conv_sme_packed_cache.kernelWidth = kernelWidth;
                conv_sme_packed_cache.packedCopies = packedCopies;
            } else if (packedKernelNeedsBuild) {
                free(packedKernel);
            }
#else
            free(packedKernel);
#endif
            return;
        }
    }
    }
#endif

#if defined(__ARM_FEATURE_SVE)
    const size_t vectorLength = svcntw();
    const size_t outputBlock = (size_t)CONV_SVE_VECTORS * vectorLength;
    const svbool_t allLanes = svptrue_b32();

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 512 && \
    CONV_SVE_VERTICAL
    const CONVINT rowTile = 4;
    const CONVINT fullOutputHeight = outputHeight - outputHeight % rowTile;
    const size_t verticalOutputBlock = 3 * vectorLength;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (CONVINT j = 0; j < fullOutputHeight; j += rowTile) {
            size_t i = 0;

            for (; i + verticalOutputBlock <= (size_t)outputWidth;
                 i += verticalOutputBlock) {
                svfloat32_t accum00 = svdup_f32(0.0f);
                svfloat32_t accum01 = svdup_f32(0.0f);
                svfloat32_t accum02 = svdup_f32(0.0f);
                svfloat32_t accum10 = svdup_f32(0.0f);
                svfloat32_t accum11 = svdup_f32(0.0f);
                svfloat32_t accum12 = svdup_f32(0.0f);
                svfloat32_t accum20 = svdup_f32(0.0f);
                svfloat32_t accum21 = svdup_f32(0.0f);
                svfloat32_t accum22 = svdup_f32(0.0f);
                svfloat32_t accum30 = svdup_f32(0.0f);
                svfloat32_t accum31 = svdup_f32(0.0f);
                svfloat32_t accum32 = svdup_f32(0.0f);

                // physicalRow 递增时，每个输出行看到的 kernelRow 也严格递增。
                for (CONVINT physicalRow = 0;
                     physicalRow < kernelHeight + rowTile - 1; ++physicalRow) {
                    const CONVFLOAT* const inputRow =
                        input + (size_t)(j + physicalRow) * inputWidth + i;

                    if (physicalRow >= rowTile - 1 && physicalRow < kernelHeight) {
                        const CONVFLOAT* const kernelRow0 =
                            kernel + (size_t)physicalRow * kernelWidth;
                        const CONVFLOAT* const kernelRow1 = kernelRow0 - kernelWidth;
                        const CONVFLOAT* const kernelRow2 = kernelRow1 - kernelWidth;
                        const CONVFLOAT* const kernelRow3 = kernelRow2 - kernelWidth;

                        for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                            const CONVFLOAT* const inputBase = inputRow + ik;
                            const svfloat32_t input0 =
                                svld1_f32(allLanes, inputBase);
                            const svfloat32_t input1 =
                                svld1_f32(allLanes, inputBase + vectorLength);
                            const svfloat32_t input2 =
                                svld1_f32(allLanes, inputBase + 2 * vectorLength);
                            const CONVFLOAT kernelValue0 = kernelRow0[ik];
                            const CONVFLOAT kernelValue1 = kernelRow1[ik];
                            const CONVFLOAT kernelValue2 = kernelRow2[ik];
                            const CONVFLOAT kernelValue3 = kernelRow3[ik];

                            accum00 = svmla_n_f32_m(allLanes, accum00, input0,
                                                    kernelValue0);
                            accum01 = svmla_n_f32_m(allLanes, accum01, input1,
                                                    kernelValue0);
                            accum02 = svmla_n_f32_m(allLanes, accum02, input2,
                                                    kernelValue0);
                            accum10 = svmla_n_f32_m(allLanes, accum10, input0,
                                                    kernelValue1);
                            accum11 = svmla_n_f32_m(allLanes, accum11, input1,
                                                    kernelValue1);
                            accum12 = svmla_n_f32_m(allLanes, accum12, input2,
                                                    kernelValue1);
                            accum20 = svmla_n_f32_m(allLanes, accum20, input0,
                                                    kernelValue2);
                            accum21 = svmla_n_f32_m(allLanes, accum21, input1,
                                                    kernelValue2);
                            accum22 = svmla_n_f32_m(allLanes, accum22, input2,
                                                    kernelValue2);
                            accum30 = svmla_n_f32_m(allLanes, accum30, input0,
                                                    kernelValue3);
                            accum31 = svmla_n_f32_m(allLanes, accum31, input1,
                                                    kernelValue3);
                            accum32 = svmla_n_f32_m(allLanes, accum32, input2,
                                                    kernelValue3);
                        }
                        continue;
                    }

                    const int active0 = physicalRow < kernelHeight;
                    const int active1 = physicalRow >= 1 && physicalRow - 1 < kernelHeight;
                    const int active2 = physicalRow >= 2 && physicalRow - 2 < kernelHeight;
                    const int active3 = physicalRow >= 3 && physicalRow - 3 < kernelHeight;
                    const CONVFLOAT* const kernelRow0 =
                        active0 ? kernel + (size_t)physicalRow * kernelWidth : NULL;
                    const CONVFLOAT* const kernelRow1 =
                        active1 ? kernel + (size_t)(physicalRow - 1) * kernelWidth : NULL;
                    const CONVFLOAT* const kernelRow2 =
                        active2 ? kernel + (size_t)(physicalRow - 2) * kernelWidth : NULL;
                    const CONVFLOAT* const kernelRow3 =
                        active3 ? kernel + (size_t)(physicalRow - 3) * kernelWidth : NULL;

                    for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                        const CONVFLOAT* const inputBase = inputRow + ik;
                        const svfloat32_t input0 =
                            svld1_f32(allLanes, inputBase);
                        const svfloat32_t input1 =
                            svld1_f32(allLanes, inputBase + vectorLength);
                        const svfloat32_t input2 =
                            svld1_f32(allLanes, inputBase + 2 * vectorLength);

                        if (active0) {
                            const CONVFLOAT kernelValue = kernelRow0[ik];
                            accum00 = svmla_n_f32_m(allLanes, accum00, input0, kernelValue);
                            accum01 = svmla_n_f32_m(allLanes, accum01, input1, kernelValue);
                            accum02 = svmla_n_f32_m(allLanes, accum02, input2, kernelValue);
                        }
                        if (active1) {
                            const CONVFLOAT kernelValue = kernelRow1[ik];
                            accum10 = svmla_n_f32_m(allLanes, accum10, input0, kernelValue);
                            accum11 = svmla_n_f32_m(allLanes, accum11, input1, kernelValue);
                            accum12 = svmla_n_f32_m(allLanes, accum12, input2, kernelValue);
                        }
                        if (active2) {
                            const CONVFLOAT kernelValue = kernelRow2[ik];
                            accum20 = svmla_n_f32_m(allLanes, accum20, input0, kernelValue);
                            accum21 = svmla_n_f32_m(allLanes, accum21, input1, kernelValue);
                            accum22 = svmla_n_f32_m(allLanes, accum22, input2, kernelValue);
                        }
                        if (active3) {
                            const CONVFLOAT kernelValue = kernelRow3[ik];
                            accum30 = svmla_n_f32_m(allLanes, accum30, input0, kernelValue);
                            accum31 = svmla_n_f32_m(allLanes, accum31, input1, kernelValue);
                            accum32 = svmla_n_f32_m(allLanes, accum32, input2, kernelValue);
                        }
                    }
                }

                CONVFLOAT* const outputRow0 = output + (size_t)j * outputWidth + i;
                CONVFLOAT* const outputRow1 = outputRow0 + outputWidth;
                CONVFLOAT* const outputRow2 = outputRow1 + outputWidth;
                CONVFLOAT* const outputRow3 = outputRow2 + outputWidth;
                svst1_f32(allLanes, outputRow0, accum00);
                svst1_f32(allLanes, outputRow0 + vectorLength, accum01);
                svst1_f32(allLanes, outputRow0 + 2 * vectorLength, accum02);
                svst1_f32(allLanes, outputRow1, accum10);
                svst1_f32(allLanes, outputRow1 + vectorLength, accum11);
                svst1_f32(allLanes, outputRow1 + 2 * vectorLength, accum12);
                svst1_f32(allLanes, outputRow2, accum20);
                svst1_f32(allLanes, outputRow2 + vectorLength, accum21);
                svst1_f32(allLanes, outputRow2 + 2 * vectorLength, accum22);
                svst1_f32(allLanes, outputRow3, accum30);
                svst1_f32(allLanes, outputRow3 + vectorLength, accum31);
                svst1_f32(allLanes, outputRow3 + 2 * vectorLength, accum32);
            }

            // 水平尾部不足 48 个输出，使用谓词 SVE 逐向量完成。
            for (CONVINT rowOffset = 0; rowOffset < rowTile; ++rowOffset) {
                CONVFLOAT* const outputRow =
                    output + (size_t)(j + rowOffset) * outputWidth;
                for (size_t tail = i; tail < (size_t)outputWidth; tail += vectorLength) {
                    const svbool_t activeLanes =
                        svwhilelt_b32((uint64_t)tail, (uint64_t)(size_t)outputWidth);
                    svfloat32_t accum = svdup_f32(0.0f);
                    for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                        const CONVFLOAT* const inputRow =
                            input + (size_t)(j + rowOffset + jk) * inputWidth + tail;
                        const CONVFLOAT* const kernelRow =
                            kernel + (size_t)jk * kernelWidth;
                        for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                            accum = svmla_n_f32_m(
                                activeLanes, accum,
                                svld1_f32(activeLanes, inputRow + ik), kernelRow[ik]);
                        }
                    }
                    svst1_f32(activeLanes, outputRow + tail, accum);
                }
            }
        }

        // 输出高度不足 4 行的尾部沿用通用谓词 SVE 路径。
#pragma omp for schedule(static)
        for (CONVINT j = fullOutputHeight; j < outputHeight; ++j) {
            CONVFLOAT* const outputRow = output + (size_t)j * outputWidth;
            for (size_t i = 0; i < (size_t)outputWidth; i += vectorLength) {
                const svbool_t activeLanes =
                    svwhilelt_b32((uint64_t)i, (uint64_t)(size_t)outputWidth);
                svfloat32_t accum = svdup_f32(0.0f);
                for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                    const CONVFLOAT* const inputRow =
                        input + (size_t)(j + jk) * inputWidth + i;
                    const CONVFLOAT* const kernelRow = kernel + (size_t)jk * kernelWidth;
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                        accum = svmla_n_f32_m(activeLanes, accum,
                                              svld1_f32(activeLanes, inputRow + ik),
                                              kernelRow[ik]);
                    }
                }
                svst1_f32(activeLanes, outputRow + i, accum);
            }
        }
    }
    return;
#endif

#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVFLOAT* const outputRow = output + (size_t)j * outputWidth;
        size_t i = 0;

        for (; i + outputBlock <= (size_t)outputWidth; i += outputBlock) {
            svfloat32_t accum0 = svdup_f32(0.0f);
            svfloat32_t accum1 = svdup_f32(0.0f);
            svfloat32_t accum2 = svdup_f32(0.0f);
#if CONV_SVE_VECTORS >= 4
            svfloat32_t accum3 = svdup_f32(0.0f);
#endif
#if CONV_SVE_VECTORS >= 5
            svfloat32_t accum4 = svdup_f32(0.0f);
#endif
#if CONV_SVE_VECTORS >= 6
            svfloat32_t accum5 = svdup_f32(0.0f);
#endif

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 512 && \
    CONV_SVE_VECTORS == 6 && CONV_SVE_SLIDING
            const svbool_t extensionLanes = svwhilelt_b32((uint64_t)0, (uint64_t)15);

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + (size_t)jk * kernelWidth;
                CONVINT ik = 0;

                for (; ik + 15 < kernelWidth; ik += 16) {
                    const CONVFLOAT* const inputBase = inputRow + ik;
                    const svfloat32_t input0 =
                        svld1_f32(allLanes, inputBase);
                    const svfloat32_t input1 =
                        svld1_f32(allLanes, inputBase + vectorLength);
                    const svfloat32_t input2 =
                        svld1_f32(allLanes, inputBase + 2 * vectorLength);
                    const svfloat32_t input3 =
                        svld1_f32(allLanes, inputBase + 3 * vectorLength);
                    const svfloat32_t input4 =
                        svld1_f32(allLanes, inputBase + 4 * vectorLength);
                    const svfloat32_t input5 =
                        svld1_f32(allLanes, inputBase + 5 * vectorLength);
                    // 最后一向量只有前 15 lane 会被 svext 使用，谓词加载避免越界。
                    const svfloat32_t input6 =
                        svld1_f32(extensionLanes, inputBase + 6 * vectorLength);

                    const CONVFLOAT kernelValue0 = kernelRow[ik];
                    accum0 = svmla_n_f32_m(allLanes, accum0, input0, kernelValue0);
                    accum1 = svmla_n_f32_m(allLanes, accum1, input1, kernelValue0);
                    accum2 = svmla_n_f32_m(allLanes, accum2, input2, kernelValue0);
                    accum3 = svmla_n_f32_m(allLanes, accum3, input3, kernelValue0);
                    accum4 = svmla_n_f32_m(allLanes, accum4, input4, kernelValue0);
                    accum5 = svmla_n_f32_m(allLanes, accum5, input5, kernelValue0);

#define CONV_SVE_SLIDE_STEP(SHIFT)                                                     \
                    do {                                                               \
                        const CONVFLOAT kernelValue = kernelRow[ik + (SHIFT)];          \
                        accum0 = svmla_n_f32_m(                                         \
                            allLanes, accum0, svext_f32(input0, input1, (SHIFT)),        \
                            kernelValue);                                               \
                        accum1 = svmla_n_f32_m(                                         \
                            allLanes, accum1, svext_f32(input1, input2, (SHIFT)),        \
                            kernelValue);                                               \
                        accum2 = svmla_n_f32_m(                                         \
                            allLanes, accum2, svext_f32(input2, input3, (SHIFT)),        \
                            kernelValue);                                               \
                        accum3 = svmla_n_f32_m(                                         \
                            allLanes, accum3, svext_f32(input3, input4, (SHIFT)),        \
                            kernelValue);                                               \
                        accum4 = svmla_n_f32_m(                                         \
                            allLanes, accum4, svext_f32(input4, input5, (SHIFT)),        \
                            kernelValue);                                               \
                        accum5 = svmla_n_f32_m(                                         \
                            allLanes, accum5, svext_f32(input5, input6, (SHIFT)),        \
                            kernelValue);                                               \
                    } while (0)

                    CONV_SVE_SLIDE_STEP(1);
                    CONV_SVE_SLIDE_STEP(2);
                    CONV_SVE_SLIDE_STEP(3);
                    CONV_SVE_SLIDE_STEP(4);
                    CONV_SVE_SLIDE_STEP(5);
                    CONV_SVE_SLIDE_STEP(6);
                    CONV_SVE_SLIDE_STEP(7);
                    CONV_SVE_SLIDE_STEP(8);
                    CONV_SVE_SLIDE_STEP(9);
                    CONV_SVE_SLIDE_STEP(10);
                    CONV_SVE_SLIDE_STEP(11);
                    CONV_SVE_SLIDE_STEP(12);
                    CONV_SVE_SLIDE_STEP(13);
                    CONV_SVE_SLIDE_STEP(14);
                    CONV_SVE_SLIDE_STEP(15);
#undef CONV_SVE_SLIDE_STEP
                }

                // 不足 16 个核列的部分继续直接加载，保持通用尺寸和原始累加顺序。
                for (; ik < kernelWidth; ++ik) {
                    const CONVFLOAT kernelValue = kernelRow[ik];
                    const CONVFLOAT* const inputBase = inputRow + ik;
                    accum0 = svmla_n_f32_m(allLanes, accum0,
                                           svld1_f32(allLanes, inputBase), kernelValue);
                    accum1 = svmla_n_f32_m(allLanes, accum1,
                                           svld1_f32(allLanes, inputBase + vectorLength),
                                           kernelValue);
                    accum2 = svmla_n_f32_m(allLanes, accum2,
                                           svld1_f32(allLanes, inputBase + 2 * vectorLength),
                                           kernelValue);
                    accum3 = svmla_n_f32_m(allLanes, accum3,
                                           svld1_f32(allLanes, inputBase + 3 * vectorLength),
                                           kernelValue);
                    accum4 = svmla_n_f32_m(allLanes, accum4,
                                           svld1_f32(allLanes, inputBase + 4 * vectorLength),
                                           kernelValue);
                    accum5 = svmla_n_f32_m(allLanes, accum5,
                                           svld1_f32(allLanes, inputBase + 5 * vectorLength),
                                           kernelValue);
                }
            }
#else
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + (size_t)jk * kernelWidth;

                CONVINT ik = 0;
                for (; ik + 1 < kernelWidth; ik += 2) {
                    const CONVFLOAT kernelValue0 = kernelRow[ik];
                    const CONVFLOAT kernelValue1 = kernelRow[ik + 1];
                    const CONVFLOAT* const inputBase = inputRow + ik;

                    accum0 = svmla_n_f32_m(allLanes, accum0,
                                           svld1_f32(allLanes, inputBase), kernelValue0);
                    accum0 = svmla_n_f32_m(allLanes, accum0,
                                           svld1_f32(allLanes, inputBase + 1), kernelValue1);
                    accum1 = svmla_n_f32_m(allLanes, accum1,
                                           svld1_f32(allLanes, inputBase + vectorLength),
                                           kernelValue0);
                    accum1 = svmla_n_f32_m(allLanes, accum1,
                                           svld1_f32(allLanes, inputBase + vectorLength + 1),
                                           kernelValue1);
                    accum2 = svmla_n_f32_m(allLanes, accum2,
                                           svld1_f32(allLanes, inputBase + 2 * vectorLength),
                                           kernelValue0);
                    accum2 = svmla_n_f32_m(allLanes, accum2,
                                           svld1_f32(allLanes, inputBase + 2 * vectorLength + 1),
                                           kernelValue1);
#if CONV_SVE_VECTORS >= 4
                    accum3 = svmla_n_f32_m(allLanes, accum3,
                                           svld1_f32(allLanes, inputBase + 3 * vectorLength),
                                           kernelValue0);
                    accum3 = svmla_n_f32_m(allLanes, accum3,
                                           svld1_f32(allLanes, inputBase + 3 * vectorLength + 1),
                                           kernelValue1);
#endif
#if CONV_SVE_VECTORS >= 5
                    accum4 = svmla_n_f32_m(allLanes, accum4,
                                           svld1_f32(allLanes, inputBase + 4 * vectorLength),
                                           kernelValue0);
                    accum4 = svmla_n_f32_m(allLanes, accum4,
                                           svld1_f32(allLanes, inputBase + 4 * vectorLength + 1),
                                           kernelValue1);
#endif
#if CONV_SVE_VECTORS >= 6
                    accum5 = svmla_n_f32_m(allLanes, accum5,
                                           svld1_f32(allLanes, inputBase + 5 * vectorLength),
                                           kernelValue0);
                    accum5 = svmla_n_f32_m(allLanes, accum5,
                                           svld1_f32(allLanes, inputBase + 5 * vectorLength + 1),
                                           kernelValue1);
#endif
                }

                if (ik < kernelWidth) {
                    const CONVFLOAT kernelValue = kernelRow[ik];
                    const CONVFLOAT* const inputBase = inputRow + ik;

                    accum0 = svmla_n_f32_m(allLanes, accum0,
                                           svld1_f32(allLanes, inputBase), kernelValue);
                    accum1 = svmla_n_f32_m(allLanes, accum1,
                                           svld1_f32(allLanes, inputBase + vectorLength),
                                           kernelValue);
                    accum2 = svmla_n_f32_m(allLanes, accum2,
                                           svld1_f32(allLanes, inputBase + 2 * vectorLength),
                                           kernelValue);
#if CONV_SVE_VECTORS >= 4
                    accum3 = svmla_n_f32_m(allLanes, accum3,
                                           svld1_f32(allLanes, inputBase + 3 * vectorLength),
                                           kernelValue);
#endif
#if CONV_SVE_VECTORS >= 5
                    accum4 = svmla_n_f32_m(allLanes, accum4,
                                           svld1_f32(allLanes, inputBase + 4 * vectorLength),
                                           kernelValue);
#endif
#if CONV_SVE_VECTORS >= 6
                    accum5 = svmla_n_f32_m(allLanes, accum5,
                                           svld1_f32(allLanes, inputBase + 5 * vectorLength),
                                           kernelValue);
#endif
                }
            }
#endif

            svst1_f32(allLanes, outputRow + i, accum0);
            svst1_f32(allLanes, outputRow + i + vectorLength, accum1);
            svst1_f32(allLanes, outputRow + i + 2 * vectorLength, accum2);
#if CONV_SVE_VECTORS >= 4
            svst1_f32(allLanes, outputRow + i + 3 * vectorLength, accum3);
#endif
#if CONV_SVE_VECTORS >= 5
            svst1_f32(allLanes, outputRow + i + 4 * vectorLength, accum4);
#endif
#if CONV_SVE_VECTORS >= 6
            svst1_f32(allLanes, outputRow + i + 5 * vectorLength, accum5);
#endif
        }

        // SVE 谓词尾部最多处理一个向量，不需要大块标量回退。
        for (; i < (size_t)outputWidth; i += vectorLength) {
            const svbool_t activeLanes =
                svwhilelt_b32((uint64_t)i, (uint64_t)(size_t)outputWidth);
            svfloat32_t accum = svdup_f32(0.0f);

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + (size_t)jk * kernelWidth;

                CONVINT ik = 0;
                for (; ik + 1 < kernelWidth; ik += 2) {
                    const CONVFLOAT* const inputBase = inputRow + ik;
                    accum = svmla_n_f32_m(activeLanes, accum,
                                          svld1_f32(activeLanes, inputBase),
                                          kernelRow[ik]);
                    accum = svmla_n_f32_m(activeLanes, accum,
                                          svld1_f32(activeLanes, inputBase + 1),
                                          kernelRow[ik + 1]);
                }
                if (ik < kernelWidth) {
                    accum = svmla_n_f32_m(activeLanes, accum,
                                          svld1_f32(activeLanes, inputRow + ik),
                                          kernelRow[ik]);
                }
            }

            svst1_f32(activeLanes, outputRow + i, accum);
        }
    }
#else
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVFLOAT* const outputRow = output + (size_t)j * outputWidth;
        CONVINT i = 0;

        for (; i + CONV_OUTPUT_BLOCK <= outputWidth; i += CONV_OUTPUT_BLOCK) {
            CONVFLOAT accum[CONV_OUTPUT_BLOCK] = {0};

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + (size_t)jk * kernelWidth;

                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const CONVFLOAT kernelValue = kernelRow[ik];

#pragma omp simd
                    for (CONVINT lane = 0; lane < CONV_OUTPUT_BLOCK; ++lane) {
                        accum[lane] += inputRow[ik + lane] * kernelValue;
                    }
                }
            }

#pragma omp simd
            for (CONVINT lane = 0; lane < CONV_OUTPUT_BLOCK; ++lane) {
                outputRow[i + lane] = accum[lane];
            }
        }

        // 处理不能组成完整向量块的输出列，累加顺序与参考实现完全一致。
        for (; i < outputWidth; ++i) {
            outputRow[i] = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + (size_t)jk * kernelWidth;

                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    outputRow[i] += inputRow[ik] * kernelRow[ik];
                }
            }
        }
    }
#endif
}

#undef CONV_OUTPUT_BLOCK
#undef CONV_SVE_VECTORS
#undef CONV_SVE_SLIDING
#undef CONV_SVE_VERTICAL
#undef CONV_USE_SME
#undef CONV_SME_ITERATIVE_WINDOWS
#undef CONV_SME_CARRY_WINDOWS
#undef CONV_SME_ALIGNED_PACKING
#undef CONV_SME_CACHE_PACKED_KERNEL
#undef CONV_SME_PACKED_KERNEL_COPIES
#undef CONV_SME_PACKED_KERNEL_COPY_MIN_BYTES
#undef CONV_SME_BLOCKS_PER_TASK
#undef CONV_SME_FINE_TASK_MIN_PER_THREAD
#undef CONV_SME_INPUT_PREFETCH_DISTANCE
#undef CONV_SME_CARRY_TAIL
#undef CONV_SME_CARRY_TAIL_MAX_WIDTH
