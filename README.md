# 鲲鹏高性能计算全球挑战赛 CONV 优化仓库

本仓库用于鲲鹏高性能计算全球挑战赛（S2 赛季）的 CONV 优化赛题。

仓库同时保留一份不可修改的初始赛题代码和一份用于实际优化的工作副本，便于随时进行源码对照、正确性回归和性能基线比较。赛事约束及仓库工作规范见根目录的 [`AGENTS.md`](./AGENTS.md)。

## 仓库结构

```text
Conv/
├── .gitignore             # 忽略本地 ZGEMM 参考目录
├── AGENTS.md              # 赛事约束及开发、验证、提交规范
├── README.md              # 仓库总览（本文件）
├── docs/
│   ├── optimization-report.md  # 完整实验记录与性能分析
│   ├── 2026-08-27-Conv高性能计算优化流程和结论.md  # 提交前精简总结
│   ├── 2026-08-27-Conv高性能计算优化流程和结论.pdf  # 提交前精简 PDF 版本
│   └── 2026-08-28-Conv下一阶段高性能优化开发计划.md  # 后续优化路线
├── conv_init/             # 官方初版备份，只读，不进行优化修改
│   ├── README.md          # 原始 CONV 赛题说明
│   ├── bench_conv.c       # 原始测试与性能评测程序
│   └── conv2d.c           # 原始待优化实现
└── conv/                  # 实际优化工作目录
    ├── README.md          # CONV 赛题说明副本
    ├── bench_conv.c       # 测试程序副本，仍然禁止修改
    ├── conv2d.c           # 实际进行性能优化的核心代码
    └── run.sh             # 官方环境一键编译和运行脚本
```

后续如需加入提交脚本、测试记录或优化说明，应放在 `conv/` 中或根目录下新建用途明确的目录，不得污染 `conv_init/`。

当前优化过程、正确性验证和性能数据见 [`docs/optimization-report.md`](./docs/optimization-report.md)。本轮 SME 优化的 AI 判断、成功路径、失败实验、瓶颈分析、扩展能力和后续方向见 [`docs/2026-08-27-Conv高性能计算优化流程和结论.md`](./docs/2026-08-27-Conv高性能计算优化流程和结论.md)。

## 目录用途

### `conv_init/`：初版备份

`conv_init/` 保存整理仓库时的原始文件，用于：

- 恢复和核对初始 `conv2d` 实现；
- 对照测试程序是否被意外修改；
- 建立优化前的性能基线；
- 审查工作副本与初版之间的差异。

该目录视为只读。除非明确要求更新官方基线，否则不得修改、格式化、删除或覆盖其中的文件。

### `conv/`：优化工作副本

所有实际优化均在 `conv/` 中完成：

- 主要修改目标为 `conv/conv2d.c` 中的 `conv2d` 函数；
- `conv/bench_conv.c` 是测试程序，不得修改；
- `conv/run.sh` 使用官方鲲鹏环境验证过的编译参数完成编译和四项测试；
- 每次优化都必须先通过正确性验证，再比较性能。

仓库整理完成时，`conv/` 与 `conv_init/` 中的三份初始文件内容完全一致。之后两者出现的差异应只来自 `conv/` 中经过验证的优化及提交辅助文件。

## 参考编译与运行

进入工作目录：

```sh
cd conv
```

参考编译命令：

```sh
gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp
```

官方公开测试用例：

```sh
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4096 6144 39 39 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6144 4096 41 41 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4256 6390 55 55 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6390 4256 81 81 1
```

评测限制为单一 NUMA 节点、最多 38 个 CPU 核心。完整规则、正确性要求和提交注意事项以 [`AGENTS.md`](./AGENTS.md) 及官方最新通知为准。

在支持 512 位 SVE/SME 的官方鲲鹏环境中，可直接执行：

```sh
cd conv
bash run.sh
```

`run.sh` 默认加载 BiSheng 5.0.0.2，以 `-O3` 编译未修改的测试程序，并仅对 `conv2d.c` 启用 `-ffast-math`。主路径使用 512 位 SVE/SME、四个 ZA tile、卷积核打包、逐 lane 滚动窗口和跨核列组窗口复用；中等 kernel 的大任务采用等重 64 列 static 任务，宽 kernel 保留 coarse + guided 调度，小任务自动回退低开销分区。GCC 模式保留显式 SVE 回退。默认绑定 NUMA 7，编译器和节点均可显式切换：

```sh
COMPILER=bisheng NUMA_NODE=7 bash run.sh
COMPILER=gcc NUMA_NODE=7 bash run.sh
```

## 建议工作流程

1. 使用 `conv_init/` 建立并保存初始性能基线。
2. 仅在 `conv/conv2d.c` 中实施优化。
3. 使用未修改的 `conv/bench_conv.c` 验证正确性。
4. 在相同编译器、线程数和 NUMA 绑定下重复测试并比较性能。
5. 检查 `conv/` 与 `conv_init/` 的差异，确认测试程序和赛题说明未被改动。
6. 使用 `conv/run.sh` 完成一键编译和测试验证，再制作提交包。
