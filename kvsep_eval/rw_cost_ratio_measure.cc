#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <chrono>

using namespace std;

// read a block with direct I/O
double read_block_direct_io(const char* filepath, size_t block_size) {
    // Open file with O_DIRECT flag for direct I/O
    int fd = open(filepath, O_RDONLY | O_DIRECT);
    if (fd == -1) {
        std::cerr << "Error opening file for direct I/O" << std::endl;
        return -1;
    }

    // Allocate aligned buffer for direct I/O
    void* buffer;
    posix_memalign(&buffer, block_size, block_size);

    // Read the block
    auto start = std::chrono::high_resolution_clock::now();
    ssize_t bytes_read = read(fd, buffer, block_size);
    if (bytes_read == -1) {
        std::cerr << "Error reading file" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Clean up
    free(buffer);
    close(fd);

    return std::chrono::duration<double, std::micro>(end - start).count();
}

double write_block_direct_io(const char* filepath, size_t block_size) {
    // Open file with O_DIRECT flag for direct I/O
    int fd = open(filepath, O_WRONLY | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        std::cerr << "Error opening file for direct I/O" << std::endl;
        return -1;
    }

    // Allocate aligned buffer for direct I/O
    void* buffer;
    posix_memalign(&buffer, block_size, block_size);
    memset(buffer, 'A', block_size); // Fill buffer with data

    // Write the block
    auto start = std::chrono::high_resolution_clock::now();
    ssize_t bytes_written = write(fd, buffer, block_size);
    if (bytes_written == -1) {
        std::cerr << "Error writing file" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Clean up
    free(buffer);
    close(fd);

    return std::chrono::duration<double, std::micro>(end - start).count();
}

double scan_direct_io(const char* filepath, size_t block_size, size_t num_blocks) {
    // prepare the test file
    int fd_prep = open(filepath, O_WRONLY | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR);
    if (fd_prep == -1) {
        std::cerr << "Error opening file for direct I/O" << std::endl;
        return -1;
    }
    // Allocate aligned buffer for direct I/O
    void* buffer_prep;
    posix_memalign(&buffer_prep, block_size, block_size);
    memset(buffer_prep, 'A', block_size); // Fill buffer with data
    for (size_t i = 0; i < num_blocks; ++i) {
        ssize_t bytes_written = write(fd_prep, buffer_prep, block_size);
        if (bytes_written == -1) {
            std::cerr << "Error writing file during preparation" << std::endl;
            break;
        }
    }
    free(buffer_prep);
    close(fd_prep);
    // Open file with O_DIRECT flag for direct I/O
    int fd = open(filepath, O_RDONLY | O_DIRECT);
    if (fd == -1) {
        std::cerr << "Error opening file for direct I/O" << std::endl;
        return -1;
    }
    // Allocate aligned buffer for direct I/O
    void* buffer;
    posix_memalign(&buffer, block_size, block_size);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_blocks; ++i) {
        ssize_t bytes_read = read(fd, buffer, block_size);
        if (bytes_read == -1) {
            std::cerr << "Error reading file" << std::endl;
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Clean up
    free(buffer);
    close(fd);

    return std::chrono::duration<double, std::micro>(end - start).count();
}

int main() {
    const char* filepath = "testfile.dat";
    size_t block_size = 4096; // 4KB block size

    // average over 10 iterations
    int iterations = 10;
    int block_count = 4;
    double total_read_time = 0.0;
    double total_write_time = 0.0;
    double total_scan_time = 0.0;
    for (int i = 0; i < iterations; ++i) {
        total_write_time += write_block_direct_io(filepath, block_size);
        total_read_time += read_block_direct_io(filepath, block_size);
        total_scan_time += scan_direct_io(filepath, block_size, block_count);
    }

    std::cout << "Average Direct I/O Read Time: " << (total_read_time / iterations) << " microseconds" << std::endl;
    std::cout << "Average Direct I/O Write Time: " << (total_write_time / iterations) << " microseconds" << std::endl;
    std::cout << "Average Direct I/O Scan Time: " << (total_scan_time / iterations / block_count) << " microseconds" << std::endl;

    return 0;
}