from numba import cuda
import numpy as np
import time


# Hàm chạy trên GPU
@cuda.jit
def counter_gpu(arr):
    idx = cuda.grid(1)  # Lấy chỉ số của thread
    if idx < arr.size:
        arr[idx] += 1  # Mỗi thread xử lý một phần tử


def main():
    num_elements = 250_000_000  # Số vòng lặp (250 triệu)
    arr = np.zeros(num_elements, dtype=np.int32)  # Mảng số 0

    # Cấu hình GPU
    threads_per_block = 1024  # Số thread mỗi block
    blocks_per_grid = (
        num_elements + (threads_per_block - 1)
    ) // threads_per_block  # Số block cần thiết

    # Chạy trên GPU
    start = time.perf_counter()
    counter_gpu[blocks_per_grid, threads_per_block](arr)  # Khởi chạy kernel trên GPU
    cuda.synchronize()  # Đồng bộ GPU và CPU
    end = time.perf_counter()

    print("PROCESSING on GPU")
    print("done in:", int(end - start), "Seconds")


if __name__ == "__main__":
    main()
