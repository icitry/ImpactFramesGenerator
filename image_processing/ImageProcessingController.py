import cv2
import numpy as np


class ImageProcessingController:
    def _dodge_v2(self, x, y):
        return cv2.divide(x, 255 - y, scale=256.0)

    def _get_sketch_impact_frame(self, frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

        edges = cv2.Canny(blurred, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.ones_like(frame) * 255

        for contour in contours:
            for point in contour:
                x, y = point[0]
                theta = np.arctan2(y, x)
                length = np.random.uniform(1, 15)  # Varying line lengths
                width = np.random.randint(1, 3)  # Varying line widths
                color_variation = np.random.randint(-255, -150)  # Random color variation
                color = int(255 + color_variation)
                x1 = int(x + length * np.cos(theta))
                y1 = int(y + length * np.sin(theta))
                x2 = int(x - length * np.cos(theta))
                y2 = int(y - length * np.sin(theta))
                cv2.line(contour_image, (x1, y1), (x2, y2), (color, color, color), width)

        return contour_image

    def _get_watercolor_impact_frame(self, frame, strength=1.2):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        details = cv2.medianBlur(gray, 11)

        combined = cv2.addWeighted(smoothed, 1.0 - strength, details, strength, 0)

        edges = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        watercolor = cv2.bitwise_and(frame, edges)

        hsvImg = cv2.cvtColor(watercolor, cv2.COLOR_BGR2HSV)

        hsvImg[..., 1] = hsvImg[..., 1] * 2

        image = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

        return image

    def _get_black_and_white_abstraction_frame(self, frame, threshold=128):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.zeros_like(frame)

        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        result = cv2.addWeighted(frame, 0.7, contour_image, 0.3, 0)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        return cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

    def _get_intermediate_binary_blur_frame(self, frame,
                                            dark_gray_intensity=80,
                                            light_gray_intensity=180,
                                            kernel_size=8):
        new_frame = frame.copy()
        black_mask = (new_frame == 0)
        white_mask = (new_frame == 255)

        new_frame[black_mask] = dark_gray_intensity

        new_frame[white_mask] = light_gray_intensity

        new_frame = cv2.blur(new_frame, (kernel_size, kernel_size))

        return new_frame

    def get_impact_frames(self, frame):
        sketch = self._get_sketch_impact_frame(frame)
        sketch_inverted = cv2.bitwise_not(sketch)
        watercolor = self._get_watercolor_impact_frame(frame)
        bw_abstract_frame = self._get_black_and_white_abstraction_frame(frame)
        bw_abstract_frame_inverted = cv2.bitwise_not(bw_abstract_frame)
        intermediate_gray_blur = self._get_intermediate_binary_blur_frame(bw_abstract_frame)
        intermediate_gray_blur_inverted = self._get_intermediate_binary_blur_frame(bw_abstract_frame_inverted)

        return [sketch,
                sketch_inverted,
                bw_abstract_frame,
                intermediate_gray_blur,
                watercolor,
                intermediate_gray_blur_inverted,
                bw_abstract_frame_inverted,
                sketch_inverted,
                sketch]
