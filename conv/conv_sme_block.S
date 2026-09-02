.text
.arch armv9-a+sme
.align 4
.global conv_sme_asm_block
.type conv_sme_asm_block, %function
conv_sme_asm_block:
    mov x8, x0
    mov x9, x2
    ptrue p0.s
    mov z0.s, #0
    ld1w { z1.s }, p0/z, [x8]
    add x2, x8, #64
    ld1w { z2.s }, p0/z, [x2]
    add x2, x8, #128
    ld1w { z3.s }, p0/z, [x2]
    add x2, x8, #192
    ld1w { z4.s }, p0/z, [x2]
    add x2, x8, #256
    ld1w { z5.s }, p0/z, [x2]
1:
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    ld1w { z6.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z6.s, z1.s
    fmopa za1.s, p0/m, p0/m, z6.s, z2.s
    fmopa za2.s, p0/m, p0/m, z6.s, z3.s
    fmopa za3.s, p0/m, p0/m, z6.s, z4.s
    ld1w { z7.s }, p0/z, [x1]
    add x1, x1, #64
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    fmopa za0.s, p0/m, p0/m, z7.s, z1.s
    fmopa za1.s, p0/m, p0/m, z7.s, z2.s
    fmopa za2.s, p0/m, p0/m, z7.s, z3.s
    fmopa za3.s, p0/m, p0/m, z7.s, z4.s
    subs x9, x9, #1
    beq 2f
    ext z1.b, z1.b, z2.b, #4
    ext z2.b, z2.b, z3.b, #4
    ext z3.b, z3.b, z4.b, #4
    ext z4.b, z4.b, z5.b, #4
    ext z5.b, z5.b, z0.b, #4
    add x8, x8, #64
    add x2, x8, #256
    ld1w { z5.s }, p0/z, [x2]
    b 1b
2:
    ret
.size conv_sme_asm_block, .-conv_sme_asm_block
