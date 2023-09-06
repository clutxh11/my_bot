#!/usr/bin/env python3

import rclpy
from my_bot.module_to_import import CannyNode, OpenCvCannyNode


def main(args=None):
    rclpy.init(args=args)

    canny_node = CannyNode()

    rclpy.spin(canny_node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
