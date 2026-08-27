# 鲲鹏高性能计算全球挑战赛（S2赛季）CONV 优化赛题

## 编译
``` shell
gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp
```

## 运行
本次四个测试用例，运行方法如下：
``` shell
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4096  6144  39 39 1 #参数依次是 rows cols kernel_rows kernel_cols test_runs表示重复测试次数，使性能稳定即可
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6144  4096  41 41 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4256  6390  55 55 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6390  4256  81 81 1

```

校验结果无误，按性能（GFLOPS）排名。
