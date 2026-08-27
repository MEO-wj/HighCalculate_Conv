#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

// ==================== 参赛者需要实现的函数 ====================
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth, CONVFLOAT* output);

// ==================== 辅助函数 ====================
// 生成随机数，算法同 std::minstd_rand0
CONVFLOAT rand_lcg(int* localSeed)
{
    int high = *localSeed / 127773;
    int low = *localSeed % 127773;
    int test = 16807 * low - 2836 * high;
    int value = test + ((test > 0) ? 0 : 2147483647);
    *localSeed = value;
    return (CONVFLOAT)value / (CONVFLOAT)2147483647.0;
}

// 生成随机浮点矩阵
void random_fmatrix(CONVFLOAT* mat, CONVINT rows, CONVINT cols, CONVINT lds, int seed)
{
#pragma omp parallel
    {
        int localSeed = seed + omp_get_thread_num();
#pragma omp for
        for (CONVINT i = 0; i < rows; ++i) {
            for (CONVINT j = 0; j < cols; ++j) {
                mat[i * lds + j] = rand_lcg(&localSeed);
            }
        }
    }
}

// 参考实现
void reference_conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
                      const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth,
                      CONVFLOAT* output)
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

// 结果校验函数
double validate_result(const CONVFLOAT* C_result, const CONVFLOAT* C_reference,
                       CONVINT outputHeight, CONVINT outputWidth, CONVINT kernelHeight,
                       CONVINT kernelWidth, double tol, int num_report)
{
    const CONVINT inputHeight = outputHeight + kernelHeight + 1;
    const CONVINT inputWidth = inputWidth + kernelWidth - 1;
    double max_error = 0.0;
    int error_count = 0;
    for (CONVINT j = 0; j < outputHeight; ++j) {
        for (CONVINT i = 0; i < outputWidth; ++i) {
            CONVINT idx = j * outputWidth + i;
            double err = fabs((double)C_result[idx] - (double)C_reference[idx]);
            if (err > tol || isnan(err)) {
                if ((++error_count) <= num_report) {
                    printf(" Error at [%d][%d]: got %.6e, expected %.6e "
                           "diff = %.6e \n",
                           j, i, C_result[idx], C_reference[idx], err);
                }
            }
            max_error = (max_error < err || isnan(err) ? err : max_error);
        }
    }
    return max_error;
}

// 性能测试函数
void test_performance(CONVINT inputHeight, CONVINT inputWidth, CONVINT kernelHeight,
                      CONVINT kernelWidth, int test_runs)
{
    printf("\n================================= CONV测试 ================================\n");
    printf("%12s %12s %16s %12s %16s %12s\n", "ImageSize", "kernelSize", "Time(ms)", "GFLOPS",
           "最大误差", "校验");
    printf("---------------------------------------------------------------------------\n");
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;
    CONVFLOAT* input = (CONVFLOAT*)malloc(inputHeight * inputWidth * sizeof(CONVFLOAT));
    CONVFLOAT* kernel = (CONVFLOAT*)malloc(kernelHeight * kernelWidth * sizeof(CONVFLOAT));
    CONVFLOAT* output = (CONVFLOAT*)malloc(outputHeight * outputWidth * sizeof(CONVFLOAT));
    CONVFLOAT* output_ref = (CONVFLOAT*)malloc(outputHeight * outputWidth * sizeof(CONVFLOAT));
    CONVFLOAT* output_check = (CONVFLOAT*)malloc(outputHeight * outputWidth * sizeof(CONVFLOAT));

    if (input == NULL || kernel == NULL || output == NULL || output_ref == NULL ||
        output_check == NULL) {
        printf("内存分配失败\n");
        free(input);
        free(kernel);
        free(output);
        free(output_ref);
        free(output_check);
        return;
    }

    // 生成测试数据
    random_fmatrix(input, inputHeight, inputWidth, inputWidth, 12345);
    random_fmatrix(kernel, kernelHeight, kernelWidth, kernelWidth, 67890);
    random_fmatrix(output, outputHeight, outputWidth, outputWidth, 13579);

    // 复制到参考矩阵
    memcpy(output_ref, output, outputHeight * outputWidth * sizeof(CONVFLOAT));
    memcpy(output_check, output, outputHeight * outputWidth * sizeof(CONVFLOAT));

    // 计算参考结果
    reference_conv2d(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_ref);

    // 运行待测试的实现
    conv2d(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_check);

    // 验证结果
    double tol = 1e-5;
    double max_err = validate_result(output_check, output_ref, outputHeight, outputWidth,
                                     kernelHeight, kernelWidth, tol, 10);

    if (max_err > tol || isnan(max_err)) {
        printf("%6d x%6d %6d x%6d %12s %12s %12.2e %8s\n", inputHeight, inputWidth, kernelHeight,
               kernelWidth, "/", "/", max_err, "FAIL");
        free(input);
        free(kernel);
        free(output);
        free(output_ref);
        free(output_check);
        return;
    }

    free(output_check);
    free(output_ref);

    // 预热
    int warmup_runs = 1;
    for (int run = 0; run < warmup_runs; ++run) {
        // 复制原始output矩阵
        CONVFLOAT* output_copy = (CONVFLOAT*)malloc(outputHeight * outputWidth * sizeof(CONVFLOAT));
        memcpy(output_copy, output, outputHeight * outputWidth * sizeof(CONVFLOAT));
        conv2d(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_copy);
        free(output_copy);
    }

    // 正式测试
    double total_time = 0.0;
    for (int run = 0; run < test_runs; ++run) {
        // 每次使用相同的输入数据（避免缓存影响）
        CONVFLOAT* output_copy = (CONVFLOAT*)malloc(outputHeight * outputWidth * sizeof(CONVFLOAT));
        memcpy(output_copy, output, outputHeight * outputWidth * sizeof(CONVFLOAT));

        double start_time = omp_get_wtime();
        conv2d(input, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, output_copy);
        double end_time = omp_get_wtime();

        total_time += (end_time - start_time);
        free(output_copy);
    }

    double avg_time = total_time / test_runs;
    double total_flops = (double)outputHeight * outputWidth * kernelHeight * kernelWidth * 2;
    double gflops = total_flops / avg_time * 1e-9;
    double avg_time_ms = avg_time * 1e3;

    printf("%6d x%6d %6d x%6d %12.2f %12.4f %12.2e %8s\n", inputHeight, inputWidth, kernelHeight,
           kernelWidth, avg_time_ms, gflops, max_err, "PASS");

    free(input);
    free(kernel);
    free(output);
}

// ==================== 主测试函数 ====================
int main(int argc, char* argv[])
{
    // 测试参数
    CONVINT inputHeight = 1024;
    CONVINT inputWidth = 1024;
    CONVINT kernelHeight = 3;
    CONVINT kernelWidth = 3;

    int test_runs = 5;

    // 从命令行读取参数
    if (argc >= 5) {
        inputHeight = atoi(argv[1]);
        inputWidth = atoi(argv[2]);
        kernelHeight = atoi(argv[3]);
        kernelWidth = atoi(argv[4]);
    }
    if (argc >= 6) {
        test_runs = atoi(argv[5]);
    }

    test_performance(inputHeight, inputWidth, kernelHeight, kernelWidth, test_runs);

    return 0;
}
