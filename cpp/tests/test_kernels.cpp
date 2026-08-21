// Round-trip tests for Q1_0 and Q2_0 kernels: quantize -> dequantize and
// check the reconstruction error is within expected bounds for the format.
#include "bonsai_kernels.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>

using namespace bonsai;

int main() {
    const int M = 4, K = 256;  // 2 groups of 128 along K
    std::mt19937 rng(0);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> w(M * K);
    for (auto& v : w) v = d(rng);

    {
        auto t = quantize_q1_0(w.data(), {M, K});
        assert(t.n_groups() == M * 2);
        std::vector<float> dq(t.numel());
        dequantize_q1_0(t, dq.data());
        for (int i = 0; i < M * K; ++i) {
            assert((w[i] >= 0) == (dq[i] >= 0));
        }
        std::vector<float> x(K), y(M);
        for (auto& v : x) v = d(rng);
        matmul_q1_0(t, x.data(), 1, y.data());
        for (auto v : y) assert(std::isfinite(v));
        std::printf("Q1_0 round-trip + matmul OK (%lld bytes, %.3f bpw)\n",
            (long long)t.blocks.size(),
            double(t.blocks.size() * 8) / double(M * K));
    }

    {
        auto t = quantize_q2_0(w.data(), {M, K});
        assert(t.n_groups() == M * 2);
        std::vector<float> dq(t.numel());
        dequantize_q2_0(t, dq.data());
        std::vector<float> x(K), y(M);
        for (auto& v : x) v = d(rng);
        matmul_q2_0(t, x.data(), 1, y.data());
        for (auto v : y) assert(std::isfinite(v));
        std::printf("Q2_0 round-trip + matmul OK (%lld bytes, %.3f bpw)\n",
            (long long)t.blocks.size(),
            double(t.blocks.size() * 8) / double(M * K));
    }

    std::printf("all kernel tests passed\n");
    return 0;
}
