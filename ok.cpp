#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    // Replace 'path/to/your/image.jpg' with the actual path to your image file
    std::string image_path = "testImgs/VKivk.jpg";

    // Load the image using OpenCV
    Mat image = imread(image_path);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        return -1;
    }

    // Access the pixel data using pointers
    uchar* data = image.data;
    int channels = image.channels();

    // Print the first 10 pixel values
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < channels; ++j) {
            std::cout << (int)data[i * channels + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}