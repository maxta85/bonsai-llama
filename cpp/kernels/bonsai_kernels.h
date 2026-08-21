// bonsai_kernels: packed 1-bit (Q1_0) and ternary 1.58-bit (Q2_0) weight
// dequantization + reference matmul kernels.
//
// These mirror the Python bonsai_tensor library and the ggml Q1_0_g128 /
// Q2_0_g128 block layouts used by the PrismML-Eng/llama.cpp fork.
//
// Block layouts (group size 128):
//   Q1_0: [fp16 scale (2 bytes)][16 bytes packed bits]  = 18 bytes/block
//   Q2_0: [fp16 scale (2 bytes)][32 bytes packed 2-bit] = 34 bytes/block
#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <string>

namespace bonsai {

constexpr int kGroupSize = 128;

// A packed Q1_0 (1-bit, g128) tensor.
struct Q1_0Tensor {
    std::vector<uint8_t> blocks;   // n_groups * 18 bytes, [scale(2)][bits(16)] per block
    std::vector<int64_t> shape;    // logical shape
    int64_t numel() const;
    int64_t n_groups() const { return static_cast<int64_t>(blocks.size() / 18); }
};

// A packed Q2_0 (ternary, g128) tensor.
struct Q2_0Tensor {
    std::vector<uint8_t> blocks;   // n_groups * 34 bytes, [scale(2)][codes(32)] per block
    std::vector<int64_t> shape;    // logical shape
    int64_t numel() const;
    int64_t n_groups() const { return static_cast<int64_t>(blocks.size() / 34); }
};

// Dequantize a Q1_0 tensor to FP32. Output size == numel().
void dequantize_q1_0(const Q1_0Tensor& t, float* out);

// Dequantize a Q2_0 tensor to FP32. Output size == numel().
void dequantize_q2_0(const Q2_0Tensor& t, float* out);

// Reference matmul: y = dequant(W) @ x.
// W shape: (M, K), x shape: (K, N), y shape: (M, N). Row-major.
void matmul_q1_0(const Q1_0Tensor& W, const float* x, int N, float* y);
void matmul_q2_0(const Q2_0Tensor& W, const float* x, int N, float* y);

// Quantize an FP32 weight buffer to Q1_0 / Q2_0 (abs-mean scaling, g128).
Q1_0Tensor quantize_q1_0(const float* w, const std::vector<int64_t>& shape);
Q2_0Tensor quantize_q2_0(const float* w, const std::vector<int64_t>& shape);

// Load a raw packed tensor from a file (header + blocks). For testing.
bool load_q1_0(const std::string& path, Q1_0Tensor& out);
bool load_q2_0(const std::string& path, Q2_0Tensor& out);

}  // namespace bonsai
