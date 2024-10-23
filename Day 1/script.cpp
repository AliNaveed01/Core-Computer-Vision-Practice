#include <opencv2/opencv.hpp>
#include <iostream>

// Function to detect kicker using white pixel coverage
bool detect_kicker_by_white_pixels(const std::string &image_path, double threshold_percentage = 5.0, int threshold_value = 127) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return false;
    }

    cv::Mat img_bw;
    cv::threshold(img, img_bw, threshold_value, 255, cv::THRESH_BINARY);

    int height = img.rows;
    int width = img.cols;

    // Analyze the upper half for kicker detection
    cv::Mat upper_region = img_bw(cv::Rect(0, 0, width, height / 2));
    int total_pixels = upper_region.total();
    int white_pixels = cv::countNonZero(upper_region);
    double white_pixel_percentage = (white_pixels / (double)total_pixels) * 100;

    std::cout << "White Pixel Percentage in Upper Region: " << white_pixel_percentage << "%" << std::endl;

    return (white_pixel_percentage > threshold_percentage);
}

// Function to check kicker orientation using pixel-based comparison
bool check_kicker_orientation(const cv::Mat &current_roi, const std::vector<std::string> &valid_roi_paths) {
    std::vector<cv::Mat> valid_rois;
    
    // Load the reference ROIs
    for (const auto &path : valid_roi_paths) {
        cv::Mat roi = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (roi.empty()) {
            std::cerr << "Error: Failed to load reference ROI: " << path << std::endl;
            return false;
        }
        valid_rois.push_back(roi);
    }

    // Compare pixel-wise difference between current ROI and valid ROIs
    std::vector<double> scores;
    for (const auto &valid_roi : valid_rois) {
        cv::Mat diff;
        cv::absdiff(current_roi, valid_roi, diff);
        double score = cv::sum(diff)[0];  // Summing the pixel differences
        scores.push_back(score);
    }

    // Find the best match based on the lowest difference score
    auto min_score_iter = std::min_element(scores.begin(), scores.end());
    int best_match_index = std::distance(scores.begin(), min_score_iter);
    
    std::cout << "Best match score: " << *min_score_iter << std::endl;

    return (best_match_index == 0 || best_match_index == 1);  // Assuming valid matches are indexed 0 or 1
}

// Main kicker pipeline function
void kicker_pipeline(const std::string &image_path, const std::vector<std::string> &valid_roi_paths) {
    if (!detect_kicker_by_white_pixels(image_path)) {
        std::cout << "No kicker detected based on white pixel percentage." << std::endl;
        return;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Image not found for orientation checking." << std::endl;
        return;
    }

    // Extract the upper half of the image for orientation checking
    cv::Mat current_roi = img(cv::Rect(0, 0, img.cols, img.rows / 2));
    
    if (check_kicker_orientation(current_roi, valid_roi_paths)) {
        std::cout << "Kicker is in a valid position." << std::endl;
    } else {
        std::cout << "Kicker is in an invalid position." << std::endl;
    }
}

int main() {
    std::string image_path = "TestImgs/Kicker at front position/Trial 1.jpg";
    std::vector<std::string> valid_roi_paths = { "Reference_Imgs/valid clean roi.jpg", "Reference_Imgs/valid noisy roi.jpg", "Reference_Imgs/invalid clean roi.jpg", "Reference_Imgs/invalid noisy roi.jpg" };

    kicker_pipeline(image_path, valid_roi_paths);

    return 0;
}
