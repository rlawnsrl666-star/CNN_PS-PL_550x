#ifndef CNN_COMMON_H
#define CNN_COMMON_H

#include <stdint.h>
#include <string.h>
#include "xparameters.h"
#include "xil_types.h"
#include "xil_io.h"


// ===================== Global Timer =====================
// Global Timer base/regs (from xparameters.h)
#define GLOBAL_TIMER_BASEADDR XPAR_GLOBAL_TIMER_BASEADDR
#define GTIMER_LOW_REG          (GLOBAL_TIMER_BASEADDR + 0x00)  // Counter Low
#define GTIMER_HIGH_REG         (GLOBAL_TIMER_BASEADDR + 0x04)  // Counter High
#define GTIMER_CONTROL_REG      (GLOBAL_TIMER_BASEADDR + 0x08)  // Control Register
// Frequency: CPU clock / 2 on Zynq-7000 (same as SCU timer)
#define GLOBAL_TIMER_FREQ_HZ    (XPAR_CPU_CORE_CLOCK_FREQ_HZ / 2)

static inline u64 Get_Global_Time() {
u32 high_val, low_val; u64 time_val;
low_val  = Xil_In32(GTIMER_LOW_REG);    // read low first (latches high)
high_val = Xil_In32(GTIMER_HIGH_REG);   // then read latched high
time_val = ((u64)high_val << 32) | (u64)low_val;
return time_val;
}

static inline double cycles_to_us(u64 cyc) {
    return (double)cyc * 1e6 / (double)GLOBAL_TIMER_FREQ_HZ;
}


// ===================== Network shape (LeNet-1 variant) =====================
enum {
    IN_C=1,     IN_H=28,    IN_W=28,
                KH=5,       KW=5,
    C1_OUT=4,   C1_H=24,    C1_W=24,        // conv1: 28->24
                P1_H=12,    P1_W=12,        // pool1: 24->12 (k=2,s=2)
    C2_OUT=12,  C2_H=8,     C2_W=8,         // conv2: 12->8
                P2_H=4,     P2_W=4,         // pool2: 8->4  (k=2,s=2)
    FC_IN=12*4*4,           FC_OUT=10,      // 12*4*4 = 192

    IMG_SIZE = IN_C * IN_H * IN_W,
    N_TEST   = 1000
};


// ===================== Embedded arrays =====================
// conv1_arr.h  : const int8_t  conv1_w_embedded[4*1*5*5];
// conv2_arr.h  : const int8_t  conv2_w_embedded[12*4*5*5];
// fc1_arr.h    : const int8_t  fc1_w_embedded[10*192];
// images_arr.h : const uint8_t test_1000_images_embedded[1000*784];
// labels_arr.h : const uint8_t test_1000_labels_embedded[1000];
#include "conv1_arr.h"
#include "conv2_arr.h"
#include "fc1_arr.h"
#include "test_images_arr.h"
#include "test_labels_arr.h"


// ===================== PS-side kernels (int8 path) =====================

// 5x5 convolution: uint8 input × int8 weight → int32 accumulator
static inline void conv5x5_nchw_u8_i8(
    const uint8_t* in, int Cin, int H, int W,
    const int8_t*  w,  int Cout,
    int32_t* out_acc
){
    const int OH=(H-KH+1), OW=(W-KW+1);
    for (int co=0; co<Cout; ++co){
        const int8_t* wco = w + co*(Cin*KH*KW);
        for (int oh=0; oh<OH; ++oh){
            for (int ow=0; ow<OW; ++ow){
                int32_t acc=0;
                for (int ci=0; ci<Cin; ++ci){
                    const int8_t*  wci  = wco + ci*(KH*KW);
                    const uint8_t* in_c = in + ci*H*W;
                    for (int kh=0; kh<KH; ++kh){
                        for (int kw=0; kw<KW; ++kw){
                            acc += (int32_t)((int)in_c[(oh+kh)*W + (ow+kw)]) * (int32_t)wci[kh*KW + kw];
                        }
                    }
                }
                out_acc[(co*OH + oh)*OW + ow] = acc;
            }
        }
    }
}


// ReLU + quantization (/128, round-to-nearest) : int32 → uint8
static inline void quant_relu_uint8_downscale(const int32_t* acc, int n, uint8_t* out){
    for (int i=0; i<n; ++i){
        int32_t v = acc[i];
        // ReLU: clamp negative to 0
        if (v <= 0) {
            out[i] = 0;
            continue;
        }
        // Positive: divide by 128 with rounding
        int32_t q = (v + 64) >> 7;
        if (q > 255) q = 255;  // clamp
        out[i] = (uint8_t)q;
    }
}


// Max-pooling (valid window) for uint8 feature maps
static inline void maxpool_nchw_valid_u8(
    const uint8_t* in, int C, int H, int W,
    int k, int s,
    uint8_t* out, int OH, int OW
){
    for (int c = 0; c < C; ++c) {
        const uint8_t* in_c = in + c * H * W;
        uint8_t* out_c = out + c * OH * OW;

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int base_h = oh * s;
                const int base_w = ow * s;

                 // Since input ≥ 0 after ReLU, start from 0 safely
                uint8_t m = 0;

                for (int kh = 0; kh < k; ++kh) {
                    const int h = base_h + kh;
                    for (int kw = 0; kw < k; ++kw) {
                        const int w = base_w + kw;
                        const uint8_t v = in_c[h * W + w];
                        if (v > m) m = v;
                    }
                }
                out_c[oh * OW + ow] = m;
            }
        }
    }
}


// Fully-connected layer: uint8 activation × int8 weight → int32 output
static inline void fc_no_bias_u8_i8(
    const int8_t* w,   // [Cout, Cin] int8
    int Cout, int Cin,
    const uint8_t* x,  // [Cin] uint8 (post-ReLU, 0..255)
    int32_t* y_acc     // [Cout] int32 accumulators
){
    for (int o = 0; o < Cout; ++o) {
        const int8_t* wrow = w + o * Cin;
        int32_t acc = 0;
        for (int i = 0; i < Cin; ++i) {
            acc += (int32_t)wrow[i] * (int32_t)x[i];
        }
        y_acc[o] = acc;
    }
}


// Single-image inference on PS side (static workspace for bare-metal)
static inline uint8_t run_forward_uint8(
    const int8_t* c1_w, const int8_t* c2_w, const int8_t* fc_w,
    const uint8_t* x_single
){
    static int32_t c1_acc[C1_OUT*C1_H*C1_W];
    static int32_t c2_acc[C2_OUT*C2_H*C2_W];
    static int32_t logits_acc[FC_OUT];
    static uint8_t c1_feat[C1_OUT*C1_H*C1_W];
    static uint8_t p1[C1_OUT*P1_H*P1_W];
    static uint8_t c2_feat[C2_OUT*C2_H*C2_W];
    static uint8_t p2[C2_OUT*P2_H*P2_W];
    static uint8_t fc_in[FC_IN];

    // conv1: u8×i8
    conv5x5_nchw_u8_i8(x_single, IN_C, IN_H, IN_W, c1_w, C1_OUT, c1_acc);
    quant_relu_uint8_downscale(c1_acc, C1_OUT*C1_H*C1_W, c1_feat);
    maxpool_nchw_valid_u8(c1_feat, C1_OUT, C1_H, C1_W, 2, 2, p1, P1_H, P1_W);

    // conv2: u8×i8
    conv5x5_nchw_u8_i8(p1, C1_OUT, P1_H, P1_W, c2_w, C2_OUT, c2_acc);
    quant_relu_uint8_downscale(c2_acc, C2_OUT*C2_H*C2_W, c2_feat);
    maxpool_nchw_valid_u8(c2_feat, C2_OUT, C2_H, C2_W, 2, 2, p2, P2_H, P2_W);

    // Flatten the pooled feature maps
    {
        int dst = 0;
        for (int c = 0; c < C2_OUT; ++c) {
            const uint8_t* src = p2 + c*(P2_H*P2_W);
            memcpy(&fc_in[dst], src, P2_H*P2_W);
            dst += P2_H*P2_W;
        }
    }

    // FC: uint8 × int8 → int32
    fc_no_bias_u8_i8(fc_w, FC_OUT, FC_IN, fc_in, logits_acc);

    // Argmax
    int best = logits_acc[0]; uint8_t best_idx=0;
    for (int i = 1; i < FC_OUT; ++i)
        if (logits_acc[i] > best) { best = logits_acc[i]; best_idx = i; }
    return best_idx;
}


// Wrapper for PS inference
static inline uint8_t ps_forward_one(
    const uint8_t* x,
    const int8_t* c1_w, const int8_t* c2_w, const int8_t* fc_w
){
    return run_forward_uint8(c1_w, c2_w, fc_w, x);
}

#endif