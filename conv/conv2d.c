#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

// 核心实现：标准四重循环
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth, CONVFLOAT* output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;
    memset(output, 0, outputHeight * outputWidth * sizeof(CONVFLOAT));
#pragma omp parallel for
    for (CONVINT j = 0; j < outputHeight; ++j) {
        for (CONVINT i = 0; i < outputWidth; ++i) {
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONVINT inputIdx = (j + jk) * inputWidth + (i + ik);
                    CONVINT kernelIdx = jk * kernelWidth + ik;
                    output[j * outputWidth + i] += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
    }
}
