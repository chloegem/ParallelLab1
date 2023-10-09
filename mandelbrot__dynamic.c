#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int width = 800; // Image width
const int height = 800; // Image height
const double xmin = -2.0;
const double xmax = 2.0; // Updated xmax
const double ymin = -2.0;
const double ymax = 2.0; // Updated ymax
const int max_iter = 256; // Updated max_iter

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c, int max_iter) {
    int count = 0;
    struct complex z;
    double temp, lengthsq;

    z.real = 0.0;
    z.imag = 0.0;

    while (count < max_iter) {
        temp = z.real * z.real - z.imag * z.imag + c.real;
        z.imag = 2.0 * z.real * z.imag + c.imag;
        z.real = temp;
        lengthsq = z.real * z.real + z.imag * z.imag;

        if (lengthsq >= 4.0) {
            return count; // Escaped
        }

        count++;
    }

    return max_iter; // Didn't escape
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;

    double start_time, end_time;
    double communication_time = 0.0;

    // Allocate memory for a portion of the image
    unsigned char* data = (unsigned char*)malloc(3 * width * (end_row - start_row));

    start_time = MPI_Wtime(); // Start the timer

    // Calculate the Mandelbrot set for the assigned rows
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < width; j++) {
            double x = xmin + (xmax - xmin) * j / (width - 1);
            double y = ymin + (ymax - ymin) * i / (height - 1);

            struct complex c;
            c.real = x;
            c.imag = y;

            int color = cal_pixel(c, max_iter);

            data[3 * (i - start_row) * width + 3 * j] = color % 8 * 32;
            data[3 * (i - start_row) * width + 3 * j + 1] = color % 16 * 16;
            data[3 * (i - start_row) * width + 3 * j + 2] = color % 32 * 8;
        }
    }

    end_time = MPI_Wtime(); // Stop the timer

    // Measure communication time during the gather operation
    MPI_Barrier(MPI_COMM_WORLD);
    double communication_start = MPI_Wtime();

    // Gather data from all processes (assuming width is the same for all)
    unsigned char* global_data = NULL;
    if (rank == 0) {
        global_data = (unsigned char*)malloc(3 * width * height);
    }

    MPI_Gather(data, 3 * width * (end_row - start_row), MPI_UNSIGNED_CHAR,
               global_data, 3 * width * (end_row - start_row), MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // Process 0 saves the image to a file
    if (rank == 0) {
        FILE* file = fopen("mandelbrot_dynamic_parallel.ppm", "wb");
        fprintf(file, "P6\n%d %d\n255\n", width, height);
        fwrite(global_data, 1, 3 * width * height, file);
        fclose(file);
        free(global_data);

        // Print runtime and communication time
        printf("Runtime: %lf seconds\n", end_time - start_time);
        communication_time = MPI_Wtime() - communication_start;
        printf("Communication Time: %lf seconds\n", communication_time);
    }

    MPI_Finalize();
    free(data);

    return 0;
}
