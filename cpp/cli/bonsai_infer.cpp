// Standalone inference CLI: quantize a raw FP32 weight file to Q1_0 or Q2_0,
// then run a reference matmul against a random input and print the output
// norm. Useful for sanity-checking the kernels end to end.
#include "bonsai_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>

using namespace bonsai;

static std::vector<float> load_raw_f32(const std::string& path, int64_t& numel) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    numel = f.tellg() / sizeof(float);
    f.seekg(0);
    std::vector<float> w(numel);
    f.read(reinterpret_cast<char*>(w.data()), numel * sizeof(float));
    return w;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: bonsai_infer <q1_0|q2_0> <weights.f32> <M> <K> [N]\n"
            "  Quantizes an (M,K) FP32 weight matrix to the chosen format,\n"
            "  runs W @ x for a random x of shape (K,N) (default N=1), and\n"
            "  prints the L2 norm of the output.\n");
        return 1;
    }
    std::string fmt = argv[1];
    std::string path = argv[2];
    int64_t M = std::atoll(argv[3]);
    int64_t K = std::atoll(argv[4]);
    int N = (argc > 5) ? std::atoi(argv[5]) : 1;

    int64_t numel;
    std::vector<float> w = load_raw_f32(path, numel);
    if (numel != M * K) {
        std::fprintf(stderr, "expected %lld floats, got %lld\n",
            (long long)(M * K), (long long)numel);
        return 1;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(K * N);
    for (auto& v : x) v = dist(rng);

    std::vector<float> y(M * N);

    if (fmt == "q1_0") {
        auto t = quantize_q1_0(w.data(), {M, K});
        std::printf("Q1_0: %lld groups, %lld bytes, %.3f bpw\n",
            (long long)t.n_groups(), (long long)t.blocks.size(),
            double(t.blocks.size() * 8) / double(M * K));
        matmul_q1_0(t, x.data(), N, y.data());
    } else if (fmt == "q2_0") {
        auto t = quantize_q2_0(w.data(), {M, K});
        std::printf("Q2_0: %lld groups, %lld bytes, %.3f bpw\n",
            (long long)t.n_groups(), (long long)t.blocks.size(),
            double(t.blocks.size() * 8) / double(M * K));
        matmul_q2_0(t, x.data(), N, y.data());
    } else {
        std::fprintf(stderr, "unknown format %s\n", fmt.c_str());
        return 1;
    }

    double nrm = 0.0;
    for (double v : y) nrm += v * v;
    std::printf("output L2 norm = %.6f\n", std::sqrt(nrm));
    return 0;
}
