#include <stddef.h>
#include <omp.h>

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

// 每个块同时计算多个相邻输出。向量 lane 对应不同的输出位置，
// 因而每个输出内部仍按照 kernel 行、列的原始顺序进行浮点累加。
#ifndef CONV_OUTPUT_BLOCK
#define CONV_OUTPUT_BLOCK 32
#endif

void conv2d(const CONVFLOAT* __restrict__ input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* __restrict__ kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* __restrict__ output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

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
}

#undef CONV_OUTPUT_BLOCK
