// Q2_0 (ternary 1.58-bit, g128) dequantization + reference matmul + quantize.
#include "bonsai_kernels.h"

#include <cmath>
#include <cstring>
#include <fstream>

namespace bonsai {

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) { f = sign << 31; }
        else {
            int e = -1; do { e++; frac <<= 1; } while ((frac & 0x400) == 0);
            exp = 127 - 15 + e; frac &= 0x3ff;
            f = (sign << 31) | (exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1f) {
        f = (sign << 31) | 0x7f800000 | (frac << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float r; std::memcpy(&r, &f, sizeof(r)); return r;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t frac = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return sign;
        frac |= 0x800000;
        uint32_t shift = 14 - exp;
        uint16_t h = sign | (frac >> shift);
        if ((frac >> (shift - 1)) & 1) h++;
        return h;
    } else if (exp == 0xff - 127 + 15) {
        if (frac) return sign | 0x7c00 | (frac >> 13);
        return sign | 0x7c00;
    }
    uint16_t h = sign | (exp << 10) | (frac >> 13);
    if ((frac >> 12) & 1) h++;
    return h;
}

int64_t Q2_0Tensor::numel() const {
    int64_t n = 1; for (auto d : shape) n *= d; return n;
}

void dequantize_q2_0(const Q2_0Tensor& t, float* out) {
    const int G = t.n_groups();
    for (int64_t g = 0; g < G; ++g) {
        const uint8_t* blk = t.blocks.data() + g * 34;
        uint16_t h; std::memcpy(&h, blk, 2);
        float scale = fp16_to_fp32(h);
        const uint8_t* codes = blk + 2;
        for (int i = 0; i < kGroupSize; ++i) {
            uint8_t q = (codes[i / 4] >> ((i % 4) * 2)) & 0x3;
            out[g * kGroupSize + i] = (float(q) - 1.0f) * scale;
        }
    }
}

void matmul_q2_0(const Q2_0Tensor& W, const float* x, int N, float* y) {
    const int64_t M = W.shape[0];
    const int64_t K = W.shape[1];
    std::vector<float> Wf(W.numel());
    dequantize_q2_0(W, Wf.data());
    for (int64_t i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += Wf[i * K + k] * x[k * N + j];
            y[i * N + j] = acc;
        }
    }
}

Q2_0Tensor quantize_q2_0(const float* w, const std::vector<int64_t>& shape) {
    int64_t numel = 1; for (auto d : shape) numel *= d;
    int64_t G = numel / kGroupSize;
    Q2_0Tensor t;
    t.shape = shape;
    t.blocks.resize(G * 34, 0);
    for (int64_t g = 0; g < G; ++g) {
        const float* block = w + g * kGroupSize;
        float abssum = 0.0f;
        for (int i = 0; i < kGroupSize; ++i) abssum += std::fabs(block[i]);
        float scale = abssum / float(kGroupSize);
        uint16_t h = fp32_to_fp16(scale);
        uint8_t* blk = t.blocks.data() + g * 34;
        std::memcpy(blk, &h, 2);
        uint8_t* codes = blk + 2;
        for (int i = 0; i < kGroupSize; ++i) {
            float wn = (scale > 0.0f) ? block[i] / scale : 0.0f;
            int8_t trinary = (wn > 0.5f) ? 1 : (wn < -0.5f ? -1 : 0);
            uint8_t q = uint8_t(trinary + 1);  // {0,1,2}
            codes[i / 4] |= uint8_t(q) << ((i % 4) * 2);
        }
    }
    return t;
}

bool load_q2_0(const std::string& path, Q2_0Tensor& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int64_t n_dims;
    f.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    out.shape.resize(n_dims);
    f.read(reinterpret_cast<char*>(out.shape.data()), n_dims * sizeof(int64_t));
    int64_t numel = 1; for (auto d : out.shape) numel *= d;
    int64_t G = numel / kGroupSize;
    out.blocks.resize(G * 34);
    f.read(reinterpret_cast<char*>(out.blocks.data()), G * 34);
    return true;
}

}  // namespace bonsai
