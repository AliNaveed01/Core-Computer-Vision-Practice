import cv2
import numpy as np

def detect_doughball_by_white_regions(image_path, height_ratio=0.37, threshold_value=127, area_threshold=300):
    # Step 1: Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    height, width = image.shape

    # Step 2: Define the region of interest (ROI)
    roi_height = int(height * height_ratio)
    start_y = (height - roi_height) // 2
    roi = image[start_y:start_y + roi_height, :]

    # Step 3: Apply binary thresholding
    _, binary_image = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 4: Find the largest connected component (blob)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Step 5: Find the largest component (excluding the background, label 0)
    if num_labels > 1:  # Ensure there's at least one component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, area = stats[largest_label, cv2.CC_STAT_LEFT:cv2.CC_STAT_AREA + 1]

        # Step 6: Check area of the blob
        if area > area_threshold:
            print("Doughball detected")
            return True
    print("No doughball detected")
    return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python doughball_detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_doughball_by_white_regions(image_path)
