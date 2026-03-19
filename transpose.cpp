#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

#include <omp.h>
#include <stdio.h>
#include <sched.h>

void transpose_image(unsigned char *image_out,
                     const unsigned char *image_in,
                     size_t width,
                     size_t height,
                     size_t cpp) // channels per pixel
{
#pragma omp parallel
    {
#pragma omp single
        printf("Using %d threads.\n", omp_get_num_threads());

#pragma omp for collapse(2)
        for (size_t row = 0; row < height; ++row)
        {
            for (size_t col = 0; col < width; ++col)
            {
                size_t in_idx = (row * width + col) * cpp;
                size_t out_idx = (col * height + row) * cpp;

                for (size_t c = 0; c < cpp; ++c)
                {
                    image_out[out_idx + c] = image_in[in_idx + c];
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
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
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    if (image_out == NULL)
    {
        printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    transpose_image(image_out, image_in, width, height, cpp);
    int tmp = width;
    width = height;
    height = tmp;
    double stop = omp_get_wtime();
    printf("Time to copy: %f s\n", stop - start);

    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

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
    stbi_image_free(image_out);

    return 0;
}