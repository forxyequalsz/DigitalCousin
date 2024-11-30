import cv2

# List of image paths
image_paths = [
    "docs/calibration_images/DSC_3796.JPG",
    "docs/calibration_images/DSC_3798.JPG",
    "docs/calibration_images/DSC_3799.JPG",
    "docs/calibration_images/DSC_3800.JPG",
    "docs/calibration_images/DSC_3801.JPG",
    "docs/calibration_images/DSC_3803.JPG",
    "docs/calibration_images/DSC_3804.JPG",
    "docs/calibration_images/DSC_3805.JPG",
    "docs/calibration_images/DSC_3806.JPG",
    "docs/calibration_images/DSC_3807.JPG",
]

# Chessboard pattern size (number of inner corners)
pattern_size = (9, 6)

# Function to resize image to a specific long edge and detect corners
def detect_chessboard_corners(image_path, long_edge=2000, pattern_size=(9, 6)):
    # Read the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Calculate scale factor based on the long edge
    scale_factor = long_edge / max(height, width)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray_resized = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_resized, pattern_size, None)
    
    return ret, corners, resized_image

# Loop through images and check chessboard detection
for image_path in image_paths:
    ret, corners, resized_image = detect_chessboard_corners(image_path, long_edge=2000, pattern_size=pattern_size)
    
    if ret:
        print(f"Chessboard detected in {image_path}: {len(corners)} corners found.")
    else:
        print(f"Chessboard NOT detected in {image_path}.")

    # Optionally visualize results
    if ret:
        # Draw corners on the image
        cv2.drawChessboardCorners(resized_image, pattern_size, corners, ret)
        cv2.imshow("Chessboard Detection", resized_image)
        cv2.waitKey(500)  # Display each image for 500ms

cv2.destroyAllWindows()