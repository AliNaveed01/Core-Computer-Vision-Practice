#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <vector>
#include <cmath>

// Function to load an image using stb_image
std::vector<unsigned char> load_image(const std::string& image_path, int &width, int &height, int &channels) {
    unsigned char *data = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        exit(1);
    }

    // Copy data to a vector for ease of use
    std::vector<unsigned char> image_data(data, data + (width * height * channels));
    stbi_image_free(data); // Free the original image data after copying
    return image_data;
}

// Function to convert RGB image to grayscale
std::vector<unsigned char> convert_to_grayscale(const std::vector<unsigned char>& image_data, int width, int height, int channels) {
    std::vector<unsigned char> grayscale_image(width * height);

    for (int i = 0; i < width * height; ++i) {
        int index = i * channels; // Each pixel has 'channels' values (RGB)
        unsigned char r = image_data[index];
        unsigned char g = image_data[index + 1];
        unsigned char b = image_data[index + 2];
        unsigned char gray = static_cast<unsigned char>(0.2989 * r + 0.5870 * g + 0.1140 * b);
        grayscale_image[i] = gray;
    }

    return grayscale_image;
}

// Function to apply binary threshold to a grayscale image
std::vector<unsigned char> apply_threshold(const std::vector<unsigned char>& grayscale_image, int width, int height, unsigned char threshold_value) {
    std::vector<unsigned char> binary_image(width * height);

    for (int i = 0; i < width * height; ++i) {
        binary_image[i] = (grayscale_image[i] >= threshold_value) ? 255 : 0;
    }

    return binary_image;
}

// Function to extract a region of interest (ROI) from an image
std::vector<unsigned char> extract_roi(const std::vector<unsigned char>& image, int width, int height, int roi_x, int roi_y, int roi_width, int roi_height) {
    std::vector<unsigned char> roi(roi_width * roi_height);

    for (int y = 0; y < roi_height; ++y) {
        for (int x = 0; x < roi_width; ++x) {
            roi[y * roi_width + x] = image[(roi_y + y) * width + (roi_x + x)];
        }
    }

    return roi;
}

// Function to count white pixels in a binary image
int count_white_pixels(const std::vector<unsigned char>& binary_image, int width, int height) {
    int white_pixel_count = 0;

    for (int i = 0; i < width * height; ++i) {
        if (binary_image[i] == 255) {
            ++white_pixel_count;
        }
    }

    return white_pixel_count;
}

// Function to compare two ROIs pixel by pixel (optional if you need comparison)
double compare_rois(const std::vector<unsigned char>& roi1, const std::vector<unsigned char>& roi2, int roi_width, int roi_height) {
    double total_difference = 0.0;

    for (int i = 0; i < roi_width * roi_height; ++i) {
        total_difference += std::abs(roi1[i] - roi2[i]);
    }

    return total_difference;
}

// Main function to detect kicker by analyzing white pixels in the upper half of the image
bool detect_kicker_by_white_pixels(const std::string& image_path, double threshold_percentage = 5.0, unsigned char threshold_value = 127) {
    int width, height, channels;
    std::vector<unsigned char> image_data = load_image(image_path, width, height, channels);

    // Convert to grayscale
    std::vector<unsigned char> grayscale_image = convert_to_grayscale(image_data, width, height, channels);

    // Apply binary threshold
    std::vector<unsigned char> binary_image = apply_threshold(grayscale_image, width, height, threshold_value);

    // Extract upper half ROI
    std::vector<unsigned char> upper_roi = extract_roi(binary_image, width, height, 0, 0, width, height / 2);

    // Count white pixels in upper half
    int total_pixels = (width * height) / 2;
    int white_pixels = count_white_pixels(upper_roi, width, height / 2);

    double white_pixel_percentage = (white_pixels / (double)total_pixels) * 100;
    std::cout << "White Pixel Percentage in Upper Region: " << white_pixel_percentage << "%" << std::endl;

    return (white_pixel_percentage > threshold_percentage);
}

int main() {
    std::string image_path = "TestImgs/Invalid Case/Kicker in reverse orientation/Trial 1.jpg";
    
    if (detect_kicker_by_white_pixels(image_path)) {
        std::cout << "Kicker detected." << std::endl;
    } else {
        std::cout << "No kicker detected." << std::endl;
    }

    return 0;
}