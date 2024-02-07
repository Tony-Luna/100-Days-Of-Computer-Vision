import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize global variables
points = []
current_image = None

def click_event(event, x, y, flags, params):
    global points, current_image
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(current_image, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(current_image, str(len(points)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if len(points) == 4:
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
            cv2.destroyAllWindows()  # Close all OpenCV windows
            break
        elif key == ord('n'):
            points = []  # Reset points
            current_image = image.copy()
            cv2.imshow(window_name, current_image)
    return points


def apply_homography(test_img, template_img, test_pts, template_pts):
    H, _ = cv2.findHomography(np.array(test_pts), np.array(template_pts))
    transformed_img = cv2.warpPerspective(test_img, H, (template_img.shape[1], template_img.shape[0]))
    return transformed_img, H

def add_gradient_to_template(template_img):
    # Create a diagonal gradient from top-left to bottom-right
    height, width, channels = template_img.shape
    
    # Initialize arrays for the gradient
    # Calculate the maximum distance possible (diagonal of the image)
    max_dist = np.sqrt(height**2 + width**2)
    
    for y in range(height):
        for x in range(width):
            # Calculate the distance of each point to the top-left corner
            dist = np.sqrt(x**2 + y**2)
            ratio = dist / max_dist
            
            # Linear gradient: Red at top-left, transitioning to blue at bottom-right
            red = 255 * (1 - ratio)
            green = 255 * ratio * (1 - ratio)  # Peak at the center of the gradient
            blue = 255 * ratio
            
            template_img[y, x] = [blue, green, red]  # Assign the gradient color to the pixel
    
    return template_img


def overlay_template_on_test(test_img, template_img, template_pts, test_pts):
    # Find the direct homography to map points from the template to the test image
    H, _ = cv2.findHomography(np.array(template_pts), np.array(test_pts))
    
    # Warp the gradient template with the direct homography
    overlay_img = cv2.warpPerspective(template_img, H, (test_img.shape[1], test_img.shape[0]))
    
    # Create a mask from the non-black pixels of the warped template
    mask = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Convert mask to 3 channels
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Use the mask to blend the warped template onto the test image
    overlay_img = cv2.bitwise_and(overlay_img, mask_3ch)
    overlay_img = cv2.addWeighted(overlay_img, 0.5, cv2.bitwise_and(test_img, cv2.bitwise_not(mask_3ch)), 0.5, 0)
    
    return overlay_img


def plot_results(test_image, template_image, test_points, template_points, transformed_image, overlay_image):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Test image with keypoints
    axs[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis('off')  # Remove the axes
    for idx, point in enumerate(test_points):
        axs[0, 0].scatter(point[0], point[1], color='yellow')
        axs[0, 0].text(point[0], point[1], str(idx+1), color='white')
    axs[0, 0].set_title('Test Image Keypoints', fontweight='bold')  # Bold title

    # Template image with keypoints
    axs[0, 1].imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
    axs[0, 1].axis('off')  # Remove the axes
    for idx, point in enumerate(template_points):
        axs[0, 1].scatter(point[0], point[1], color='red')
        axs[0, 1].text(point[0], point[1], str(idx+1), color='white')
    axs[0, 1].set_title('Template Image Keypoints', fontweight='bold')  # Bold title

    # Homography of Image 1 to Template 2
    axs[1, 0].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    axs[1, 0].axis('off')  # Remove the axes
    axs[1, 0].set_title('Homography: Test to Template', fontweight='bold')  # Bold title

    # Homography of Template 2 to Image 1 with Gradient
    axs[1, 1].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    axs[1, 1].axis('off')  # Remove the axes
    axs[1, 1].set_title('Overlay: Gradient Template on Test', fontweight='bold')  # Bold title

    plt.tight_layout()
    plt.show()


def main(config):
    test_image = cv2.imread(config['test_image_path'], cv2.IMREAD_COLOR)
    template_image = cv2.imread(config['template_image_path'], cv2.IMREAD_COLOR)
    
    test_points = select_points(test_image, "Test Image")
    template_points = select_points(template_image, "Template Image")

    transformed_image, _ = apply_homography(test_image, template_image, test_points, template_points)
    gradient_template = add_gradient_to_template(template_image.copy())
    overlay_image = overlay_template_on_test(test_image, gradient_template, template_points, test_points)

    plot_results(test_image, template_image, test_points, template_points, transformed_image, overlay_image)

if __name__ == "__main__":
    config = {
        'test_image_path': 'Images/test_2.png',
        'template_image_path': 'Images/field_template.png'
    }
    
    main(config)
