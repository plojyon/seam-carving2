#include <stdio.h>
#include <cmath>
#include <bits/stdc++.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// will char work faster than double?
#define ENERGY_TYPE double
#define BIG_VALUE 10000

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

void grayscalify(ENERGY_TYPE *image, const size_t width, const size_t height, const size_t cpp)
{
    for (size_t i = 0; i < width * height; i++)
    {
        ENERGY_TYPE sum = 0;
        for (size_t c = 0; c < cpp; c++)
        {
            sum += image[cpp * i + c];
        }
        image[i] = sum / cpp;
    }
}

unsigned char *normalize(ENERGY_TYPE *energy, const size_t width, const size_t height, const size_t cpp)
{
    unsigned char *normalized = (unsigned char *)calloc(width * height * sizeof(unsigned char) * cpp, 1);
    ENERGY_TYPE max_e = 0;
    for (size_t i = 0; i < width * height; i++)
    {
        if (energy[i] > max_e)
        {
            max_e = energy[i];
        }
    }
    for (size_t i = 0; i < width * height; i++)
    {
        unsigned char r = (unsigned char)((double)energy[i] / max_e * 255);
        for (size_t c = 0; c < cpp; c++)
        {
            normalized[cpp * i + c] = c > 2 ? 255 : r;
        }
    }
    printf("%f seems like a normal number\n", max_e);
    return normalized;
}

void energy(
    ENERGY_TYPE *energy,
    const unsigned char *image,
    const int width,
    const int height,
    const int cpp,
    const int thread_count)
{
#pragma omp parallel
    {
        auto s = [&](int row, int col, int c) -> int
        {
            row = std::clamp(row, 0, height - 1);
            col = std::clamp(col, 0, width - 1);
            return image[(row * width + col) * cpp + c];
        };

        int batch_size = height / thread_count;

#pragma omp for
        for (int row_batch_offset = 0; row_batch_offset < height; row_batch_offset += batch_size)
        {
            for (int col = 0; col < width; col++)
            {
                for (int row = row_batch_offset; (row < height) & (row < row_batch_offset + batch_size); row++)
                {
                    double mag = 0;

                    for (int c = 0; c < cpp; c++)
                    {
                        double Gx = -s(row - 1, col - 1, c) - 2 * s(row, col - 1, c) - s(row + 1, col - 1, c) + s(row - 1, col + 1, c) + 2 * s(row, col + 1, c) + s(row + 1, col + 1, c);
                        double Gy = +s(row - 1, col - 1, c) + 2 * s(row - 1, col, c) + s(row - 1, col + 1, c) - s(row + 1, col - 1, c) - 2 * s(row + 1, col, c) - s(row + 1, col + 1, c);

                        mag += Gx * Gx + Gy * Gy;
                    }

                    energy[row * width + col] = (ENERGY_TYPE)mag / cpp;
                }
            }
        }
    }
}

void cum_line(ENERGY_TYPE* energy, int x_start, int x_end, const int y, const int width, const float brightness) {
    // helper for cum_parallel

    x_start = std::clamp(x_start, 0, width);
    x_end = std::clamp(x_end, 0, width);

    for (int x = x_start; x < x_end; x++) {
        ENERGY_TYPE min_e = energy[(y + 1) * width + x];
        if (x > 0) {
            min_e = std::min(min_e, energy[(y + 1) * width + x - 1]);
            }
        if (x < width - 1) {
            min_e = std::min(min_e, energy[(y + 1) * width + x + 1]);
            }
        energy[y * width + x] += min_e;

        // energy[y * width + x] = brightness * BIG_VALUE; // TODO: remove
    }
}

void cum_parallel(ENERGY_TYPE *energy, const size_t width, const size_t height, const int n_threads) {
    // parameters
    const int thread_count = n_threads; // will consume thread_count or thread_count + 1 threads

    const int trig_base = width / thread_count + (width % thread_count != 0);
    const int trig_height = trig_base / 2 + (trig_base % 2 != 0);

    const int trig_width = 2*trig_height; // must be even!! please
    const int band_count = height / (trig_height + 1) + (height % (trig_height + 1) != 0);
    const int bandwidth = trig_height + 1;

    for (int band = 0; band < band_count; band++) {
        // Keep in mind y_start > y_end
        int y_start = height - 2 - band * bandwidth;
        int y_end = y_start > bandwidth? y_start - bandwidth : -1;
        #pragma omp parallel 
    {
        #pragma omp for
        for (int tid = 0; tid < thread_count; tid++) {
            // earthbound triangles
            for (int i = 0; i < y_start - y_end; i++) {
                // const int tid = 0; //omp_get_thread_num();
                const int x_start = tid * trig_width + i;
                const int x_end = x_start + trig_width - 2 * i;
                cum_line(energy, x_start, x_end, y_start - i, width, 1.0f);
        }
    }
        #pragma omp for
        for (int tid = 0; tid < thread_count + 1; tid++) {
            // moonbound triangles
            for (int i = 0; i < y_start - y_end; i++) {
                // const int tid = 0; //omp_get_thread_num();
                const int x_start = tid * trig_width - i;
                const int x_end = x_start + 2 * i;
                cum_line(energy, x_start, x_end, y_start - i, width, 2.0f);
            }
        }
    }
}
}



void left_shift_range(unsigned char *img,
                      size_t src_start,
                      size_t src_end,
                      size_t l_shift)
{

    if (src_start >= src_end)
    {
        return;
    }

    if (l_shift == 0)
    {
        return;
    }

    if (src_start < l_shift)
    {
        fprintf(stderr,
                "Error: destination underflows. Tried to move [%zu, %zu) by %zu\n",
                src_start,
                src_end,
                l_shift);
        exit(EXIT_FAILURE);
    }

    unsigned char *src = img + src_start;
    unsigned char *dst = img + (src_start - l_shift);

    size_t pixels_to_move = src_end - src_start;
    size_t bytes_to_move = pixels_to_move;

    memmove(dst, src, bytes_to_move);
}

void remove_pixel(size_t row, size_t col, unsigned char *image, const size_t cpp, const size_t width, const size_t height, bool debug)
{
    if (!debug)
    {
        // Shift before the ray because of all accumulated black pixels
        left_shift_range(image, row * width * cpp, (row * width + col) * cpp, (row)*cpp);
        // Shift after the ray because of all accumulated black pixels + 1
        left_shift_range(image, (row * width + col + 1) * cpp, (row + 1) * width * cpp, (row + 1) * cpp);
    }
    else
    {
        // make the pixel red to visualize the seam
        for (size_t c = 0; c < cpp; c++)
        {
            image[row * width * cpp + col * cpp + c] = (c == 0) ? 255 : 0;
        }
    }
}

size_t remove_seam(ENERGY_TYPE *cum_energy, unsigned char *image, const size_t channels, const size_t width, const size_t height, bool debug)
{

    // Find smallest energy in the firt row to start
    size_t min_col_idx = 0;
    ENERGY_TYPE min_e = cum_energy[0];
    for (size_t col = 1; col < width; col++)
    {
        if (cum_energy[col] < min_e)
        {
            min_col_idx = col;
            min_e = cum_energy[col];
        }
    }
    remove_pixel(0, min_col_idx, image, channels, width, height, debug);

    for (size_t row = 1; row < height; row++)
    {
        ENERGY_TYPE min_e = cum_energy[row * width + min_col_idx];
        if (min_col_idx != 0)
        {
            if (cum_energy[row * width + min_col_idx - 1] < min_e)
            {
                min_e = cum_energy[row * width + min_col_idx - 1];
                min_col_idx--;
            }
        }
        if (min_col_idx != width - 1)
        {
            if (cum_energy[row * width + min_col_idx + 1] < min_e)
            {
                min_e = cum_energy[row * width + min_col_idx + 1];
                min_col_idx++;
            }
        }
        remove_pixel(row, min_col_idx, image, channels, width, height, debug);
    }
    return width - (not debug);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("USAGE: ./a.out input_image output_image remove_N_seams [n_threads_energy] [n_threads_cum]\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, width, height, cpp);
    const size_t datasize = width * height * cpp * sizeof(ENERGY_TYPE);
    ENERGY_TYPE *image_energy = (ENERGY_TYPE *)malloc(datasize);
    if (image_energy == NULL)
    {
        printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    int remove_N_seams;
    char *percent = strchr(argv[3], '%');
    if (percent == NULL)
    {
        remove_N_seams = atoi(argv[3]);
    }
    else
    {
        *percent = '\0';
        remove_N_seams = (int)(atoi(argv[3]) / 100.0 * width);
        printf("Removing %d seams\n", remove_N_seams);
    }

    int max_threads;

#pragma omp parallel
    {
#pragma omp single
        {
            max_threads = omp_get_num_threads();
        }
    }

    int n_threads_energy;
    if (argc < 5)
    {
        n_threads_energy = max_threads;
        printf("n_threads_energy ommited, using max threads (%d)\n", int(n_threads_energy));
    }
    else
    {
        n_threads_energy = atoi(argv[4]);
    }
    int n_threads_cum;
    if (argc < 6)
    {
        n_threads_cum = std::min(max_threads, height / 210 + 1);
        printf("n_threads_cum ommited, using heuristic (%d)\n", int(n_threads_cum));
    }
    else
    {
        n_threads_cum = atoi(argv[5]);
    }

    double energy_time_sum = 0;
    double cum_energy_time_sum = 0;
    double seam_time_sum = 0;

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    for (size_t i = 0; i < remove_N_seams; i++)
    {
        double start_iter = omp_get_wtime();
        energy(image_energy, image_in, width, height, cpp, n_threads_energy);
        double stop_energy = omp_get_wtime();
        cum_parallel(image_energy, width, height, n_threads_cum);
        double stop_cum_energy = omp_get_wtime();
        width = remove_seam(image_energy, image_in, cpp, width, height, 0); // i == remove_N_seams - 1
        double stop_seam = omp_get_wtime();

        energy_time_sum += stop_energy - start_iter;
        cum_energy_time_sum += stop_cum_energy - stop_energy;
        seam_time_sum += stop_seam - stop_cum_energy;

        if (remove_N_seams < 10 || i % (remove_N_seams / 10) == 0)
        {
            printf("\r%d%%", (i * 100 / remove_N_seams));
            fflush(stdout);
        }
    }
    double stop = omp_get_wtime();
    double total_partial_time = energy_time_sum + cum_energy_time_sum + seam_time_sum;
    if (remove_N_seams != 0) {
        printf("\nTotal: %fs, Avg iter: %f\n", stop - start, (stop - start) / remove_N_seams);
        printf("Avg energy time: %fs (%.2f%%)\n", energy_time_sum / remove_N_seams, 100 * energy_time_sum / total_partial_time);
        printf("Avg cumulative energy time: %fs (%.2f%%)\n", cum_energy_time_sum / remove_N_seams, 100 * cum_energy_time_sum / total_partial_time);
        printf("Avg seam removal time: %fs (%.2f%%)\n", seam_time_sum / remove_N_seams, 100 * seam_time_sum / total_partial_time);
    }
    if (remove_N_seams == 0) {
        energy(image_energy, image_in, width, height, cpp, n_threads_energy);
        cum_parallel(image_energy, width, height, n_threads_cum);
        width = remove_seam(image_energy, image_in, cpp, width, height, true);
    }

    unsigned char *image_out = image_in;
    // unsigned char *image_out = normalize(image_energy, width, height, cpp);

    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL)
    {
        printf("Error: No file extension found!\n");
        stbi_image_free(image_in);
        stbi_image_free(image_out);
        exit(EXIT_FAILURE);
    }
    file_type++; // skip the dot

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);
    if (image_out != image_in)
    {
        stbi_image_free(image_out);
    }
    return 0;
}
