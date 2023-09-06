#!/usr/bin/env python3

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image

class CannyNode(Node):
    def __init__(self):
        super().__init__('canny_node')
        self.get_logger().info("node started")

        # Create a subscriber to the existing compressed image topic
        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10  # QoS profile history depth
        )
        self.subscription

        # Create a publisher to publish the processed edge map
        self.publisher = self.create_publisher(
            Image,
            "/canny_edge",
            10  # QoS profile history depths
        )

    def image_callback(self, msg: Image):
        # Convert the ROS Image message to a NumPy array
        np_arr = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, -1))

        # Apply your Canny edge detection algorithm to get the processed edge map
        processed_edge_map = self.canny_algorithm(np_arr, 0, 50)

        # Then, publish the processed edge map
        self.publish_edge_map(processed_edge_map)

    
    def Grayscale(self,image):
        if len(image.shape) == 3:  # Check if color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    
    def GaussianBlur(self,image, kernel_size=(3, 3), sigma=1.0):
        if len(image.shape) == 3:  # Color image, convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a Gaussian kernel
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - kernel_size[0] // 2) ** 2 + (y - kernel_size[1] // 2) ** 2)
                / (2 * sigma**2)
            ),
            kernel_size,
        )
        kernel /= np.sum(kernel)

        # Get image dimensions
        height, width = image.shape

        # Initialize the output image
        blurred_image = np.zeros((height, width))

        # Pad the image with zeros to handle border pixels
        padded_image = np.pad(
            image,
            (
                (kernel_size[0] // 2, kernel_size[0] // 2),
                (kernel_size[1] // 2, kernel_size[1] // 2),
            ),
            mode="constant",
        )

        # Convolve the image with the Gaussian kernel
        for i in range(height):
            for j in range(width):
                blurred_image[i, j] = np.sum(
                    padded_image[i : i + kernel_size[0], j : j + kernel_size[1]] * kernel
                )

        return blurred_image.astype(np.uint8)


    def SobelFilter(self, image):
        image = self.Grayscale(self.GaussianBlur(image))
        convolved = np.zeros(image.shape)
        G_x = np.zeros(image.shape)
        G_y = np.zeros(image.shape)
        size = image.shape
        kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
        kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                G_x[i, j] = np.sum(
                    np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_x)
                )
                G_y[i, j] = np.sum(
                    np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_y)
                )

        convolved = np.sqrt(np.square(G_x) + np.square(G_y))
        convolved = np.multiply(convolved, 255.0 / convolved.max())

        angles = np.rad2deg(np.arctan2(G_y, G_x))
        angles[angles < 0] += 180
        convolved = convolved.astype("uint8")
        return convolved, angles


    def non_maximum_suppression(self, image, angles):
        size = image.shape
        suppressed = np.zeros(size)
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    value_to_compare = max(image[i, j - 1], image[i, j + 1])
                elif 22.5 <= angles[i, j] < 67.5:
                    value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
                elif 67.5 <= angles[i, j] < 112.5:
                    value_to_compare = max(image[i - 1, j], image[i + 1, j])
                else:
                    value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

                if image[i, j] >= value_to_compare:
                    suppressed[i, j] = image[i, j]
        suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
        return suppressed


    def double_threshold_hysteresis(self, image, low, high):
        weak = 50
        strong = 255
        size = image.shape
        result = np.zeros(size)
        weak_x, weak_y = np.where((image > low) & (image <= high))
        strong_x, strong_y = np.where(image >= high)
        result[strong_x, strong_y] = strong
        result[weak_x, weak_y] = weak
        dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
        dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
        size = image.shape

        while len(strong_x):
            x = strong_x[0]
            y = strong_y[0]
            strong_x = np.delete(strong_x, 0)
            strong_y = np.delete(strong_y, 0)
            for direction in range(len(dx)):
                new_x = x + dx[direction]
                new_y = y + dy[direction]
                if (new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (
                    result[new_x, new_y] == weak
                ):
                    result[new_x, new_y] = strong
                    np.append(strong_x, new_x)
                    np.append(strong_y, new_y)
        result[result != strong] = 0
        return result

    def canny_algorithm(self, image, low, high):
        image, angles = self.SobelFilter(image)
        image = self.non_maximum_suppression(image, angles)
        gradient = np.copy(image)
        image = self.double_threshold_hysteresis(image, low, high)
        return image, gradient


    def publish_edge_map(self, processed_edge_map):
        # Extract the edge map from the processed_edge_map tuple
        edge_map = processed_edge_map[0]

        # Convert the edge map to Image format
        # The edge map is a NumPy array, so we need to convert it to an OpenCV matrix
        edge_map_cv = cv2.convertScaleAbs(edge_map)

        # Create an Image message and publish it
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = edge_map.shape[0]
        msg.width = edge_map.shape[1]
        msg.encoding = "mono8"  # Assuming the edge map is a grayscale image
        msg.is_bigendian = False
        msg.step = msg.width
        msg.data = edge_map_cv.tobytes()

        self.publisher.publish(msg)


class OpenCvCannyNode(Node):
    def __init__(self):
        super().__init__('cv_canny_node')
        self.get_logger().info("Node started")

        # Create a subscriber to the existing image topic
        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",  # Replace this topic with your input image topic
            self.image_callback,
            10  # QoS profile history depth
        )
        self.subscription

        # Create a publisher to publish the processed edge map
        self.publisher = self.create_publisher(
            Image,
            "/cv_canny_edge",
            10  # QoS profile history depths
        )

    def image_callback(self, msg: Image):
        # Convert the ROS Image message to a NumPy array
        np_arr = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, -1))

        # Apply Canny edge detection algorithm
        edge_map = cv2.Canny(np_arr, 100, 200)  # Modify thresholds (100, 200) as needed

        # Convert the edge map to Image format
        edge_map_msg = Image()
        edge_map_msg.header = msg.header
        edge_map_msg.height = edge_map.shape[0]
        edge_map_msg.width = edge_map.shape[1]
        edge_map_msg.encoding = "mono8"  # Edge map is a grayscale image
        edge_map_msg.is_bigendian = False
        edge_map_msg.step = edge_map_msg.width
        edge_map_msg.data = edge_map.tobytes()

        # Publish the processed edge map
        self.publisher.publish(edge_map_msg)





