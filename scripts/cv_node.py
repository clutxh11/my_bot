#!/usr/bin/env python3

import rclpy
from my_bot.module_to_import import CannyNode, OpenCvCannyNode


def main(args=None):
    rclpy.init(args=args)

    cv_canny_node = OpenCvCannyNode()

    rclpy.spin(cv_canny_node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
