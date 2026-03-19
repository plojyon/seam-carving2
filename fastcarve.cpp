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

using namespace std;

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

void make_black(ENERGY_TYPE *image, const size_t width, const size_t height)
{
    for (size_t i = 0; i < width * height; i++)
    {
        image[i] = 0;
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

void energy(ENERGY_TYPE *energy, const unsigned char *image,
            const int width, const int max_col,
            const int height, const int cpp, const thread_count)
{
    auto s = [&](int row, int col, int c) -> int
    {
        row = std::clamp(row, 0, height - 1);
        col = std::clamp(col, 0, max_col - 1);
        return image[(row * width + col) * cpp + c];
    };

    int batch_size = height / thread_count;
#pragma omp for
    for (int row_batch_offset = 0; row_batch_offset < height; row_batch_offset += batch_size)
    {
        for (int col = 0; col < max_col; col++)
        {
            for (int row = row_batch_offset; (row < max_row) & (row < row_batch_offset + batch_size); row++)
            {
                double mag = 0;

                for (int c = 0; c < cpp; c++)
                {
                    double Gx = -s(row - 1, col - 1, c) - 2 * s(row, col - 1, c) - s(row + 1, col - 1, c) + s(row - 1, col + 1, c) + 2 * s(row, col + 1, c) + s(row + 1, col + 1, c);
                    double Gy = +s(row - 1, col - 1, c) + 2 * s(row - 1, col, c) + s(row - 1, col + 1, c) - s(row + 1, col - 1, c) - 2 * s(row + 1, col, c) - s(row + 1, col + 1, c);

                    mag += sqrt(Gx * Gx + Gy * Gy);
                }

                energy[row * width + col] = (ENERGY_TYPE)mag / cpp;
            }
        }
    }
}

void cum_energy_path_cost(ENERGY_TYPE *energy, const size_t width, const size_t max_col, const size_t height)
{
    // Calculate cumulative path cost starting from bottom to top
    for (int row = height - 2; row >= 0; row--)
    {
        for (int col = 0; col < max_col; col++)
        {
            ENERGY_TYPE min_e = energy[(row + 1) * width + col];
            if (col != 0)
            {
                min_e = min(min_e, energy[(row + 1) * width + col - 1]);
            }
            if (col != max_col - 1)
            {
                min_e = min(min_e, energy[(row + 1) * width + col + 1]);
            }
            energy[row * width + col] += min_e;
        }
    }
}

void remove_pixel(size_t row, size_t col, unsigned char *image, const size_t cpp, const size_t width, const size_t height, bool debug)
{
    if (!debug)
    {
        // energy_pixel_idx is the index of the pixel in the energy array (single channel)

        // Start of the row
        unsigned char *row_start = image + row * width * cpp;

        // Destination (pixel to remove)
        unsigned char *dst = row_start + col * cpp;

        // Source (next pixel to the right)
        unsigned char *src = row_start + (col + 1) * cpp;

        // Number of bytes to shift
        size_t bytes_to_move = (width - col - 1) * cpp;

        memmove(dst, src, bytes_to_move);

        image[(row * width + width - 1) * cpp + 0] = 0;
        image[(row * width + width - 1) * cpp + 1] = 0;
        image[(row * width + width - 1) * cpp + 2] = 0;
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

void remove_seam(ENERGY_TYPE *cum_energy, unsigned char *image, const size_t channels, const size_t width, const size_t max_col, const size_t height, bool debug)
{
    // Find smallest energy in the firt row to start
    size_t min_col_idx = 0;
    ENERGY_TYPE min_e = cum_energy[0];
    for (size_t col = 1; col < max_col; col++)
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
        if (min_col_idx != max_col - 1)
        {
            if (cum_energy[row * width + min_col_idx + 1] < min_e)
            {
                min_e = cum_energy[row * width + min_col_idx + 1];
                min_col_idx++;
            }
        }
        remove_pixel(row, min_col_idx, image, channels, width, height, debug);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("USAGE: ./a.out input_image output_image remove_N_seams\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);
    int remove_N_seams = atoi(argv[3]);

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

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    for (size_t i = 0; i < remove_N_seams; i++)
    {
        energy(image_energy, image_in, width, width - i, height, cpp);

        cum_energy_path_cost(image_energy, width, width - i, height);
        remove_seam(image_energy, image_in, cpp, width, width - i, height, 0); // i == remove_N_seams - 1

        if (i % (remove_N_seams / 10) == 0)
        {
            printf("\r%d%%", (i * 100 / remove_N_seams));
            fflush(stdout);
        }
    }
    double stop = omp_get_wtime();
    printf("\nTotal: %fs, Avg iter: %f\n", stop - start, (stop - start) / remove_N_seams);

    // make_black(image_energy, width, height);
    // energy(image_energy, image_in, width, width - 199, height, cpp);

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
