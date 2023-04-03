import cv2 as cv

from maunder.util import BaseNode

from maunder_interfaces.msg import Sighting

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


class ObjectDetectorNode(BaseNode):

    def __init__(self, node):
        super().__init__(node)
        self.sighting_pub = self.node.create_publisher(Sighting, '/sighting',
                                                       10)
        self.node.create_subscription(Image, '/image/raw', self.on_image, 10)

    def on_image(self, msg):
        pass


def main():
    rclpy.init()
    node = ObjectDetectorNode(Node('object_detector'))
    rclpy.spin(node.get_node())


if __name__ == '__main__':
    main()
