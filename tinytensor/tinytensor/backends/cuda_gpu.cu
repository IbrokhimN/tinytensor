#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

extern "C" void run_cublas_matmul(const float* host_A, const float* host_B, float* host_C, int M, int K, int N) {
    size_t bytes_a = M * K * sizeof(float);
    size_t bytes_b = K * N * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, bytes_a) != cudaSuccess) throw std::runtime_error("cuda mem alloc a failed");
    if (cudaMalloc(&d_B, bytes_b) != cudaSuccess) throw std::runtime_error("cuda mem allco B failed");
    if (cudaMalloc(&d_C, bytes_c) != cudaSuccess) throw std::runtime_error("cuda mem alloc C failed");

    cudaMemcpy(d_A, host_A, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, bytes_b, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    cudaMemcpy(host_C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
