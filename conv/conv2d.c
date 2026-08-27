#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

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

#ifndef CONV_SME_ALIGNED_PACKING
#define CONV_SME_ALIGNED_PACKING 1
#endif

#if CONV_SME_CARRY_WINDOWS && !CONV_SME_ITERATIVE_WINDOWS
#error "CONV_SME_CARRY_WINDOWS requires CONV_SME_ITERATIVE_WINDOWS"
#endif

#if defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SVE_BITS) && \
    __ARM_FEATURE_SVE_BITS == 512 && CONV_USE_SME
__arm_new("za") __arm_locally_streaming static void
conv2d_sme_worker(const CONVFLOAT* __restrict__ input, CONVINT inputWidth,
                  const CONVFLOAT* __restrict__ packedKernel,
                  CONVINT kernelHeight, CONVINT kernelWidth,
                  CONVFLOAT* __restrict__ output, CONVINT outputHeight,
                  CONVINT outputWidth, CONVINT threadId, CONVINT threadCount)
{
    const uint32_t tileSize = 16;
    /*
     * Medium kernels benefit from finer column partitions.  Wider kernels
     * already provide enough work per row tile, so two partitions retain
     * more sequential cache locality without compromising load balance.
     */
    const CONVINT tasksPerRowTile = kernelWidth > 64 ? 2 : 4;
    const CONVINT totalRowTiles = (outputHeight + (CONVINT)tileSize - 1) /
                                  (CONVINT)tileSize;
    const CONVINT fullColBlocks = outputWidth / (4 * (CONVINT)tileSize);
    const CONVINT totalTasks = totalRowTiles * tasksPerRowTile;
    const CONVINT firstTask =
        (CONVINT)(((int64_t)totalTasks * threadId) / threadCount);
    const CONVINT lastTask =
        (CONVINT)(((int64_t)totalTasks * (threadId + 1)) / threadCount);

    for (CONVINT task = firstTask; task < lastTask; ++task) {
        const CONVINT rowTileIndex = task / tasksPerRowTile;
        const CONVINT colPartition = task % tasksPerRowTile;
        const CONVINT outputRowStart = rowTileIndex * (CONVINT)tileSize;
        const CONVINT rowCount =
            outputHeight - outputRowStart < (CONVINT)tileSize
                ? outputHeight - outputRowStart
                : (CONVINT)tileSize;
        const svbool_t activeRows = svwhilelt_b32((uint64_t)0, (uint64_t)rowCount);
        const CONVINT prefetchRowsPerStep =
            (kernelWidth >= 41 && kernelWidth <= 64) ? 2 : 1;
        const CONVINT prefetchSteps =
            (rowCount + prefetchRowsPerStep - 1) / prefetchRowsPerStep;
        const CONVINT prefetchStart =
            kernelHeight + rowCount - 1 - prefetchSteps;

        const CONVINT firstColBlock =
            (CONVINT)(((int64_t)fullColBlocks * colPartition) / tasksPerRowTile);
        const CONVINT lastColBlock =
            (CONVINT)(((int64_t)fullColBlocks * (colPartition + 1)) /
                      tasksPerRowTile);
        const CONVINT fullColEnd = lastColBlock * 4 * (CONVINT)tileSize;
        CONVINT outputColStart = firstColBlock * 4 * (CONVINT)tileSize;
        for (; outputColStart < fullColEnd;
             outputColStart += 4 * (CONVINT)tileSize) {
            const svbool_t allCols = svptrue_b32();
            svzero_za();

            for (CONVINT physicalRow = 0;
                 physicalRow < kernelHeight + rowCount - 1; ++physicalRow) {
                const CONVFLOAT* const inputRow =
                    input + (size_t)(outputRowStart + physicalRow) * inputWidth +
                    outputColStart;
                const CONVFLOAT* const packedRow =
                    packedKernel + (size_t)physicalRow * kernelWidth * tileSize;
                const CONVINT prefetchStep = physicalRow - prefetchStart;
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
                const svfloat32_t zeroWindow = svdup_f32(0.0f);
                svfloat32_t window0 = zeroWindow;
                svfloat32_t window1 = zeroWindow;
                svfloat32_t window2 = zeroWindow;
                svfloat32_t window3 = zeroWindow;
                svfloat32_t window4 = zeroWindow;
#endif
                for (; ik + 15 < kernelWidth; ik += 16) {
                    const CONVFLOAT* const inputBase = inputRow + ik;
#if CONV_SME_ITERATIVE_WINDOWS && CONV_SME_CARRY_WINDOWS
                    const svbool_t finalInputLanes =
                        ik + 31 < kernelWidth ? allCols : extensionLanes;
                    if (ik == 0) {
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
                        svld1_f32(activeRows, packedRow + (size_t)ik * tileSize);
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

#define CONV_SME_SLIDE_STEP(SHIFT)                                              \
                    do {                                                        \
                        window0 = svext_f32(window0, window1, 1);               \
                        window1 = svext_f32(window1, window2, 1);               \
                        window2 = svext_f32(window2, window3, 1);               \
                        window3 = svext_f32(window3, window4, 1);               \
                        window4 = svext_f32(window4, zeroWindow, 1);            \
                        const svfloat32_t kernelValues = svld1_f32(             \
                            activeRows, packedRow + (size_t)(ik + (SHIFT)) *   \
                                                      tileSize);                \
                        svmopa_za32_f32_m(0, activeRows, allCols, kernelValues, \
                                          window0);                             \
                        svmopa_za32_f32_m(1, activeRows, allCols, kernelValues, \
                                          window1);                             \
                        svmopa_za32_f32_m(2, activeRows, allCols, kernelValues, \
                                          window2);                             \
                        svmopa_za32_f32_m(3, activeRows, allCols, kernelValues, \
                                          window3);                             \
                    } while (0)

#if !CONV_SME_CARRY_WINDOWS
#define CONV_SME_SLIDE_FINAL(SHIFT)                                             \
                    do {                                                        \
                        window0 = svext_f32(window0, window1, 1);               \
                        window1 = svext_f32(window1, window2, 1);               \
                        window2 = svext_f32(window2, window3, 1);               \
                        window3 = svext_f32(window3, window4, 1);               \
                        const svfloat32_t kernelValues = svld1_f32(             \
                            activeRows, packedRow + (size_t)(ik + (SHIFT)) *   \
                                                      tileSize);                \
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
                            activeRows, packedRow + (size_t)(ik + (SHIFT)) *       \
                                                      tileSize);                    \
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

                for (; ik < kernelWidth; ++ik) {
                    const svfloat32_t kernelValues =
                        svld1_f32(activeRows, packedRow + (size_t)ik * tileSize);
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
                        svld1_f32(activeRows, packedRow + (size_t)ik * tileSize);
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
#endif

void conv2d(const CONVFLOAT* __restrict__ input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* __restrict__ kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* __restrict__ output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#if defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SVE_BITS) && \
    __ARM_FEATURE_SVE_BITS == 512 && CONV_USE_SME
    const size_t tileSize = 16;
    const size_t packedPhysicalRows = (size_t)kernelHeight + tileSize - 1;
    size_t packedGroups = 0;

    if (kernelHeight > 0 && kernelWidth > 0 &&
        packedPhysicalRows <= SIZE_MAX / (size_t)kernelWidth) {
        packedGroups = packedPhysicalRows * (size_t)kernelWidth;
    }

    if (packedGroups > 0 && packedGroups <= SIZE_MAX / tileSize &&
        packedGroups * tileSize <= SIZE_MAX / sizeof(CONVFLOAT)) {
        const size_t packedBytes =
            packedGroups * tileSize * sizeof(CONVFLOAT);
#if CONV_SME_ALIGNED_PACKING
        /* One packed 16-lane FP32 vector is exactly one 64-byte cache line. */
        CONVFLOAT* const packedKernel =
            (CONVFLOAT*)aligned_alloc(64, packedBytes);
#else
        CONVFLOAT* const packedKernel =
            (CONVFLOAT*)malloc(packedBytes);
#endif
        if (packedKernel != NULL) {
#pragma omp parallel
            {
#pragma omp for schedule(static)
                for (size_t group = 0; group < packedGroups; ++group) {
                    const CONVINT physicalRow =
                        (CONVINT)(group / (size_t)kernelWidth);
                    const CONVINT ik = (CONVINT)(group % (size_t)kernelWidth);
                    CONVFLOAT* const packedValues = packedKernel + group * tileSize;
                    for (CONVINT outputRow = 0; outputRow < (CONVINT)tileSize;
                         ++outputRow) {
                        const CONVINT jk = physicalRow - outputRow;
                        packedValues[outputRow] =
                            (jk >= 0 && jk < kernelHeight)
                                ? kernel[(size_t)jk * kernelWidth + ik]
                                : 0.0f;
                    }
                }

                conv2d_sme_worker(input, inputWidth, packedKernel, kernelHeight,
                                  kernelWidth, output, outputHeight, outputWidth,
                                  omp_get_thread_num(), omp_get_num_threads());
            }
            free(packedKernel);
            return;
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
