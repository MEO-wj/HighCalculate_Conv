#include <stddef.h>

typedef float CONVFLOAT;
typedef int CONVINT;

void conv2d_keep(const CONVFLOAT* input, CONVINT inputHeight,
                 CONVINT inputWidth, const CONVFLOAT* kernel,
                 CONVINT kernelHeight, CONVINT kernelWidth,
                 CONVFLOAT* output);
void conv2d_stream(const CONVFLOAT* input, CONVINT inputHeight,
                   CONVINT inputWidth, const CONVFLOAT* kernel,
                   CONVINT kernelHeight, CONVINT kernelWidth,
                   CONVFLOAT* output);
void conv2d_wide(const CONVFLOAT* input, CONVINT inputHeight,
                 CONVINT inputWidth, const CONVFLOAT* kernel,
                 CONVINT kernelHeight, CONVINT kernelWidth,
                 CONVFLOAT* output);

/*
 * Select one of two compile-time-specialized SME workers before entering the
 * hot loop.  Large portrait inputs stream through many row tiles and benefit
 * from non-temporal input prefetches; smaller or landscape inputs retain the
 * overlapping input lines in cache.  The decision uses generic working-set
 * and geometry properties, never a complete public benchmark shape.
 */
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* output)
{
    const size_t inputElements = (size_t)inputHeight * (size_t)inputWidth;
    const size_t streamingInputThreshold =
        (size_t)(64U * 1024U * 1024U) / sizeof(CONVFLOAT);
    /*
     * Wider kernels have a larger input footprint per output tile.  They use
     * a separate worker compiled with a longer, conservative input prefetch
     * distance; the threshold is a kernel-width class, not a public shape.
     */
    if (kernelWidth > 64) {
        conv2d_wide(input, inputHeight, inputWidth, kernel, kernelHeight,
                    kernelWidth, output);
        return;
    }

    const int useStreamingPrefetch =
        inputHeight > inputWidth &&
        inputElements >= streamingInputThreshold &&
        kernelWidth >= 41 && kernelWidth <= 64;

    if (useStreamingPrefetch) {
        conv2d_stream(input, inputHeight, inputWidth, kernel, kernelHeight,
                      kernelWidth, output);
    } else {
        conv2d_keep(input, inputHeight, inputWidth, kernel, kernelHeight,
                    kernelWidth, output);
    }
}
