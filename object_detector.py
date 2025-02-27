import cv2


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        #mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #mask = cv2.Canny(blurred, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV)

        mask = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                cnt = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours
    
    def detect_objectsv1(self, frame):
        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Otsu's thresholding
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # Filter out small contours
                cnt = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)