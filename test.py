import cv2
import numpy as np

# Function to detect kicker using white pixel coverage
def detect_kicker_by_white_pixels(image_path, threshold_percentage=5, threshold_value=127):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return False

    _, img_bw = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    height, width = img.shape
    upper_region = img_bw[0:int(height / 2), :]  # Analyze the upper half for kicker
    total_pixels = upper_region.size
    white_pixels = np.sum(upper_region == 255)
    white_pixel_percentage = (white_pixels / total_pixels) * 100
    print(f'White Pixel Percentage in Upper Region: {white_pixel_percentage:.2f}%')

    if white_pixel_percentage > threshold_percentage:
        print("Kicker Detected by White Pixels")
        return True
    else:
        print("No Kicker Detected by White Pixels")
        return False

# Function to detect kicker based on lines and white pixel coverage
def detect_kicker(image_path, min_line_count=2, min_intersect_count=1, white_pixel_threshold_percentage=5):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None, None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    height, width = edges.shape
    roi = edges[0:int(height / 2), 0:width]
    lines = cv2.HoughLines(roi, 1, np.pi/180, 100)

    line_image = np.zeros_like(image)
    line_count = 0

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            line_count += 1

    kicker_detected = (line_count >= min_line_count)
    if kicker_detected:
        print("Lines detected, checking white pixel coverage...")
        if not detect_kicker_by_white_pixels(image_path, white_pixel_threshold_percentage):
            print("No kicker detected due to insufficient white pixels.")
            return None, None
        return roi, edges
    else:
        print("No sufficient lines detected. Checking white pixel coverage...")
        if detect_kicker_by_white_pixels(image_path, white_pixel_threshold_percentage):
            print("Kicker detected by white pixel coverage.")
            return roi, edges
        else:
            print("No kicker detected.")
            return None, None

# Extract central vertical region from an image for comparison
def extract_vertical_central_region(image, width_ratio=0.3, height_ratio=0.8):
    height, width = image.shape
    central_width = int(width * width_ratio)
    central_height = int(height * height_ratio)
    x_start = (width - central_width) // 2
    y_start = (height - central_height) // 2
    return image[y_start:y_start+central_height, x_start:x_start+central_width]

# Template matching function for kicker orientation comparison
def template_matching(img1, img2):
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val

# Feature matching using ORB for kicker orientation comparison
def feature_matching(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    score = sum([1 - (match.distance / 100) for match in matches])
    return score

# Histogram comparison for kicker orientation comparison
def histogram_comparison(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Combining different scores into one
def combined_score(template_score, feature_score, hist_score):
    return 0.4 * template_score + 0.3 * feature_score + 0.3 * hist_score

# Function to check kicker orientation based on reference images
def check_kicker_orientation(current_roi, valid_roi_paths, width_ratio=0.3, height_ratio=0.8):
    valid_rois = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in valid_roi_paths]
    if any(roi is None for roi in valid_rois):
        print("Error: One or more reference ROIs not found.")
        return False

    current_central = extract_vertical_central_region(current_roi, width_ratio, height_ratio)
    central_regions = [extract_vertical_central_region(roi, width_ratio, height_ratio) for roi in valid_rois]
    roi_descriptions = ["clean valid ROI", "noisy valid ROI", "clean invalid ROI", "noisy invalid ROI"]

    scores = []
    for i, reference_central in enumerate(central_regions):
        template_score = template_matching(current_central, reference_central)
        feature_score = feature_matching(current_central, reference_central)
        hist_score = histogram_comparison(current_central, reference_central)
        combined = combined_score(template_score, feature_score, hist_score)
        scores.append(combined)
        print(f"ROI {roi_descriptions[i]} - Combined Score: {combined:.4f}")

    best_match_index = np.argmax(scores)
    best_score = scores[best_match_index]
    threshold = 10

    if best_score > threshold:
        if best_match_index in [0, 1]:
            print(f"Kicker is in valid orientation (matched with {roi_descriptions[best_match_index]}).")
            return True
        else:
            print(f"Kicker is in invalid orientation (matched with {roi_descriptions[best_match_index]}).")
            return False
    else:
        print("No clear valid match, kicker might be in invalid orientation.")
        return False

# Full pipeline
def kicker_pipeline(image_path, valid_roi_paths):
    current_roi, current_edges = detect_kicker(image_path)
    if current_roi is None:
        print("No kicker detected.")
        return False
    return check_kicker_orientation(current_roi, valid_roi_paths)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python kicker_pipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    valid_roi_paths = [
        'Reference_Imgs/valid clean roi.jpg',
        'Reference_Imgs/valid noisy roi.jpg',
        'Reference_Imgs/invalid clean roi.jpg',
        'Reference_Imgs/invalid noisy roi.jpg'
    ]

    result = kicker_pipeline(image_path, valid_roi_paths)
    if result:
        print("Kicker in valid position.")
    else:
        print("Issue with the kicker")
