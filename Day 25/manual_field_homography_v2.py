import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize global variables
points = []
current_image = None

def click_event(event, x, y, flags, params):
    global points, current_image
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        # Append point on mouse click
        points.append((x, y))
        # Draw circle and number on the image
        cv2.circle(current_image, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(current_image, str(len(points)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if len(points) == 4:
            # Draw the message to keep or discard points
            cv2.putText(current_image, "Keep points? (y/n)", (10, current_image.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(params['window_name'], current_image)

def select_points(image, window_name):
    global points, current_image
    points = []  # Reset points
    current_image = image.copy()
    cv2.imshow(window_name, current_image)
    cv2.setMouseCallback(window_name, click_event, {'window_name': window_name})
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y'):
            break
        elif key == ord('n'):
            # Reset points and remove the (y/n) message by redrawing the image
            points = []
            current_image = image.copy()
            cv2.imshow(window_name, current_image)
    # Keep the window open for reference
    return points

def apply_homography(test_img, template_img, test_pts, template_pts):
    H, _ = cv2.findHomography(np.array(test_pts), np.array(template_pts))
    height, width, channels = template_img.shape
    transformed_img = cv2.warpPerspective(test_img, H, (width, height))
    return transformed_img, H

def plot_matching_points(test_image, template_image, test_points, template_points):
    plt.figure(figsize=(10, 5))

    # Display test image with points
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.scatter(*zip(*test_points), color='yellow', label='Test Image Points')
    plt.title('Test Image with Selected Points')
    plt.legend()

    # Display template image with points
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
    plt.scatter(*zip(*template_points), color='red', label='Template Image Points')
    plt.title('Template Image with Selected Points')
    plt.legend()

    plt.show()

def main(config):
    # Load images
    test_image = cv2.imread(config['test_image_path'], cv2.IMREAD_COLOR)
    template_image = cv2.imread(config['template_image_path'], cv2.IMREAD_COLOR)
    
    # Select points on the test image
    test_points = select_points(test_image, "Test Image")

    # Display the test image with points for reference
    for point in test_points:
        cv2.circle(test_image, point, 5, (0, 255, 0), -1)

    # Select points on the template image without closing the test image window
    template_points = select_points(template_image, "Template Image")

    # Apply homography
    transformed_image, _ = apply_homography(test_image, template_image, test_points, template_points)

    # Plot results and matching points
    plot_matching_points(test_image, template_image, test_points, template_points)

    # Display transformed image
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    config = {
        'test_image_path': 'Images/test_1.png',
        'template_image_path': 'Images/field_template.png'
    }
    
    main(config)
