#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

extern "C" void run_cublas_matmul(const float* host_A, const float* host_B, float* host_C, int M, int K, int N);

py::array_t<float> cublas_matmul(py::array_t<float> a_py, py::array_t<float> b_py) {
    auto buf_a = a_py.request();
    auto buf_b = b_py.request();

    if (buf_a.ndim != 2 || buf_b.ndim != 2) {
        throw std::runtime_error("только 2д матрицы разрешены");
    }

    int M = buf_a.shape[0];
    int K = buf_a.shape[1];
    int N = buf_b.shape[1];

    if (K != buf_b.shape[0]) {
        throw std::runtime_error("несовпадение размерностей");
    }

    auto result = py::array_t<float>({M, N});
    auto buf_c = result.request();

    run_cublas_matmul(
        static_cast<const float*>(buf_a.ptr),
        static_cast<const float*>(buf_b.ptr),
        static_cast<float*>(buf_c.ptr),
        M, K, N
    );

    return result;
}

PYBIND11_MODULE(cuda_ops, m) {
    m.doc() = "tinytensor cuBLAS backend";
    m.def("matmul", &cublas_matmul, "high performance matmul with nvidia cublas");
}