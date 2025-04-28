import cv2
import numpy as np

class HueSegmentationTool:
    """
    A tool for interactive hue-based segmentation of an image using OpenCV trackbars.

    This tool allows the user to adjust HSV thresholds dynamically and view the original
    and segmented images side-by-side in real-time.
    """

    def __init__(self, image_path: str):
        """
        Initializes the segmentation tool by loading an image and setting up trackbars.

        Args:
            image_path (str): The path to the input image.
        """
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Could not load the image: {image_path}")

        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.window_name = "Hue Segmentation"

        self._setup_trackbars()

    def _setup_trackbars(self):
        """
        Sets up HSV threshold trackbars in a dedicated OpenCV window.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 300)

        cv2.createTrackbar("Hue Min", self.window_name, 0, 179, lambda x: None)
        cv2.createTrackbar("Hue Max", self.window_name, 179, 179, lambda x: None)
        cv2.createTrackbar("Sat Min", self.window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("Sat Max", self.window_name, 255, 255, lambda x: None)
        cv2.createTrackbar("Val Min", self.window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("Val Max", self.window_name, 255, 255, lambda x: None)

    def _get_trackbar_values(self):
        """
        Retrieves the current values of the HSV trackbars.

        Returns:
            tuple: Lower and upper HSV threshold arrays.
        """
        h_min = cv2.getTrackbarPos("Hue Min", self.window_name)
        h_max = cv2.getTrackbarPos("Hue Max", self.window_name)
        s_min = cv2.getTrackbarPos("Sat Min", self.window_name)
        s_max = cv2.getTrackbarPos("Sat Max", self.window_name)
        v_min = cv2.getTrackbarPos("Val Min", self.window_name)
        v_max = cv2.getTrackbarPos("Val Max", self.window_name)
        return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

    def run(self):
        """
        Runs the main loop to display the original and segmented images side-by-side.

        Controls:
            ESC or 'q' - Exit the tool.
        """
        print("Controls:\n  ESC or 'q' = quit")

        while True:
            lower, upper = self._get_trackbar_values()

            mask = cv2.inRange(self.hsv, lower, upper)
            result = cv2.bitwise_and(self.img, self.img, mask=mask)

            img_resized = cv2.resize(self.img, (self.img.shape[1] // 2, self.img.shape[0] // 2))
            result_resized = cv2.resize(result, (result.shape[1] // 2, result.shape[0] // 2))
            combined = np.hstack((img_resized, result_resized))

            cv2.imshow(self.window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()

# Example usage:
# tool = HueSegmentationTool("path_to_your_image.jpg")
# tool.run()
