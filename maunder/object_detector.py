from math import asin, cos, pi, sin, sqrt

import cv2 as cv

from diff_drive.transformations \
    import euler_from_quaternion, quaternion_from_euler

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped

from maunder.util import BaseNode

from maunder_interfaces.msg import Sighting

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

from tf2_ros import TransformBroadcaster


class BallFinder:

    def __init__(self, min_hue=100, max_hue=120, blur=0, erode=3, dilate=3,
                 edge_dilation=2, min_saturation=125, min_value=50,
                 min_radius=8, min_circularity=0.5, min_ratio=0.6,
                 min_area=250, max_radial_distance=480, min_y=300, max_y=650):
        self.min_hue = min_hue
        self.max_hue = max_hue
        self.blur = blur
        self.erode = erode
        self.dilate = dilate
        self.edge_dilation = edge_dilation
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_radius = min_radius
        self.min_circularity = min_circularity
        self.min_ratio = min_ratio
        self.min_area = min_area
        self.max_radial_distance = max_radial_distance
        self.lower = np.array([self.min_hue, self.min_saturation,
                               self.min_value])
        self.upper = np.array([self.max_hue, 255, 255])
        self.min_y = min_y
        self.max_y = max_y

    def find_ball(self, orig):
        hsv = cv.cvtColor(orig, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower, self.upper)
        masked = cv.bitwise_and(orig, orig, mask=mask)
        gray = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
        img = gray
        if self.blur > 0:
            blur_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                   (self.blur, self.blur))
            img = cv.erode(img, blur_kernel)
        if self.erode > 0:
            erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                    (self.erode, self.erode))
            img = cv.dilate(img, erode_kernel)
        if self.dilate > 0:
            dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                     (self.dilate, self.dilate))
            img = cv.dilate(img, dilate_kernel)
        edges = cv.Canny(img, 50, 100, apertureSize=3)
        if self.edge_dilation > 0:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                              (self.edge_dilation,
                                               self.edge_dilation))
        edges = cv.dilate(edges, kernel)
        contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_SIMPLE)

        targets = []
        xc = orig.shape[1] / 2
        yc = orig.shape[0] / 2
        for i in range(len(contours)):
            center, radius = cv.minEnclosingCircle(contours[i])
            perimeter = cv.arcLength(contours[i], closed=True)
            area = cv.contourArea(contours[i])
            ratio = area/(pi * radius**2)
            circularity = 4 * pi * area / perimeter**2
            d = sqrt((center[0] - xc)**2 + (center[1] - yc)**2)
            if radius >= self.min_radius \
               and ratio >= self.min_ratio \
               and circularity >= self.min_circularity \
               and d <= self.max_radial_distance \
               and area >= self.min_area \
               and center[1] >= self.min_y \
               and center[1] <= self.max_y:
                targets.append((center, radius, area, circularity, d, ratio))
        targets.sort(key=lambda data: -data[2])
        if not targets:
            return None
        else:
            return targets[0][0], targets[0][1]


class GoalFinder:

    def __init__(self):
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)

    def find_goal(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners, ids, rej = cv.aruco.detectMarkers(gray, self.dictionary)
        if not ids:
            return None
        return self.get_detection(corners[0])

    def get_detection(self, corner):
        ul, ur, lr, ll = corner[0]
        radius = ((ur[0] - ul[0]) + (lr[0] - ll[0])) / 4
        center_x = (ur[0] + ul[0] + lr[0] + ll[0]) / 4
        center_y = (ur[1] + ul[1] + lr[1] + ll[1]) / 4
        return (center_x, center_y), radius


class ObjectDetectorNode(BaseNode):

    def __init__(self, node):
        super().__init__(node)

        min_hue = self.get_parameter('min_hue', 100)
        max_hue = self.get_parameter('max_hue', 120)
        min_saturation = self.get_parameter('min_saturation', 125)
        min_value = self.get_parameter('min_value', 50)
        min_radius = self.get_parameter('min_radius', 8)
        blur = self.get_parameter('blur', 0)
        erode = self.get_parameter('erode', 3)
        dilate = self.get_parameter('dilate', 3)
        edge_dilation = self.get_parameter('edge_dilation', 2)
        min_circularity = self.get_parameter('min_circularity', 0)
        min_ratio = self.get_parameter('min_ratio', 0.6)
        min_area = self.get_parameter('min_area', 250)
        max_radial_distance = self.get_parameter('max_radial_distance',
                                                     480)
        min_y = self.get_parameter('min_y', 300)
        max_y = self.get_parameter('max_y', 650)

        self.ball_slope = self.get_parameter('ball_slope', 11.5)
        self.ball_bias = self.get_parameter('ball_bias', 0.070)

        self.goal_slope = self.get_parameter('goal_slope', 9.13)
        self.goal_bias = self.get_parameter('goal_bias', -0.00991)

        self.focal_length = self.get_parameter('focal_length', 628.41)
        self.beta = self.get_parameter('beta', 0.635)
        self.y_max_angle = self.get_parameter('y_fov', 80.0) / 2

        self.ball_finder = BallFinder(
            min_hue=min_hue,
            max_hue=max_hue,
            blur=blur,
            erode=erode,
            dilate=dilate,
            edge_dilation=edge_dilation,
            min_saturation=min_saturation,
            min_value=min_value,
            min_radius=min_radius,
            min_circularity=min_circularity,
            min_ratio=min_ratio,
            min_area=min_area,
            max_radial_distance=max_radial_distance,
            min_y=min_y,
            max_y=max_y
        )

        self.goal_finder = GoalFinder()

        self.tf_broadcaster = TransformBroadcaster(self.node)

        self.ball_sighting_pub = self.node.create_publisher(
            Sighting, 'ball_sighting', 10)
        self.ball_point_pub = self.node.create_publisher(
            PointStamped, 'ball_image_position', 10)

        self.goal_sighting_pub = self.node.create_publisher(
            Sighting, 'goal_sighting', 10)
        self.goal_point_pub = self.node.create_publisher(
            PointStamped, 'goal_image_position', 10)

        self.node.create_subscription(Image, 'camera/image', self.on_image, 10)

    def on_image(self, msg):
        frame = np.reshape(np.frombuffer(msg.data, np.uint8),
                           (msg.height, msg.width, 3))

        ball_detection = self.ball_finder.find_ball(frame)
        if ball_detection:
            ball_center, ball_radius = ball_detection
            ball_sighting = Sighting()
            ball_sighting.heading = self.get_heading(
                ball_center[0], ball_center[1], msg.width, msg.height)
            ball_sighting.distance = self.get_ball_distance(ball_radius)
            self.ball_sighting_pub.publish(ball_sighting)
            self.send_transform(msg.header.stamp, 'ball', 'base_link',
                                ball_sighting.heading, ball_sighting.distance)

        goal_detection = self.goal_finder.find_goal(frame)
        if goal_detection:
            goal_center, goal_radius = goal_detection
            goal_sighting = Sighting()
            goal_sighting.heading = self.get_heading(
                goal_center[0], goal_center[1], msg.width, msg.height)
            goal_sighting.distance = self.get_goal_distance(goal_radius)
            self.goal_sighting_pub.publish(goal_sighting)
            self.send_transform(msg.header.stamp, 'goal', 'base_link',
                                goal_sighting.heading, goal_sighting.distance)

    def get_heading(self, x, y, img_width, img_height):
        half_x = img_width / 2
        half_y = img_height / 2
        dx = x - half_x
        dy = y - half_y
        y_angle = dy / half_y * self.y_max_angle
        x_adjusted = x
        #x_adjusted = half_x + dx * cos(y_angle)
        return asin((half_x - x_adjusted) / self.focal_length) / self.beta

    def get_ball_distance(self, radius):
        return self.ball_slope/radius + self.ball_bias

    def get_goal_distance(self, width):
        return self.goal_slope/width + self.goal_bias

    def send_transform(self, stamp, child_frame_id, frame_id,
                       heading, distance):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        q = quaternion_from_euler(0, 0, heading+pi)
        t.transform.translation.x = distance * cos(heading)
        t.transform.translation.y = distance * sin(heading)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = ObjectDetectorNode(Node('object_detector'))
    rclpy.spin(node.get_node())


if __name__ == '__main__':
    main()
