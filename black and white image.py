import cv2
import numpy as np


image_path = r'C:\Users\alway\Desktop\Learning\Deep Learning Projects\Image Processing\Black and White Image\about.jpg'


image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to open image file.")
else:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)

    # Convert back to BGR to apply further processing
    equalized_color = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    # Sharpen the image using a custom kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(equalized_color, -1, kernel)

    # Denoise the image using a bilateral filter
    denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)

    # Display the original and processed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Gray Image", gray)
    cv2.imshow("Equalized Image", equalized)
    cv2.imshow("Sharpened Image", sharpened)
    cv2.imshow("Denoised Image", denoised)

    # Save the processed images
    cv2.imwrite(r'C:\Users\alway\Desktop\Learning\Deep Learning Projects\Image Processing\Black and White Image\gray_image.jpg', gray)
    cv2.imwrite(r'C:\Users\alway\Desktop\Learning\Deep Learning Projects\Image Processing\Black and White Image\equalized_image.jpg',
                equalized)
    cv2.imwrite(r'C:\Users\alway\Desktop\Learning\Deep Learning Projects\Image Processing\Black and White Image\sharpened_image.jpg',
                sharpened)
    cv2.imwrite(r'C:\Users\alway\Desktop\Learning\Deep Learning Projects\Image Processing\Black and White Image\denoised_image.jpg', denoised)

    print("Processed images saved successfully.")

    # Wait for a key press and close all OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
