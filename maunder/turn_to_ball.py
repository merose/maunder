from geometry_msgs.msg import Twist

from maunder.util import BaseNode

from maunder_interfaces.msg import Sighting

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Access, Status
from py_trees.composites import Sequence

from py_trees_ros.publishers import FromBlackboard
from py_trees_ros.subscribers import ToBlackboard
from py_trees_ros.trees import BehaviourTree

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles


class TurnToHeading(Behaviour):

    def setup(self, **kwargs):
        self.bb = py_trees.blackboard.Client()
        self.bb.register_key(key="/parameters/kp", access=Access.READ)
        self.bb.register_key(key="/ball_sighting", access=Access.READ)
        self.bb.register_key(key="/cmd_vel", access=Access.WRITE)

    def update(self):
        heading = self.bb.ball_sighting.heading
        if abs(heading) < 0.08:
            angular = 0.0
        elif heading >= 0:
            angular = min(0.1, self.bb.parameters.kp * heading)
        else:
            angular = max(-0.1, self.bb.parameters.kp * heading)
        twist = Twist()
        twist.angular.z = angular
        self.bb.cmd_vel = twist
        return Status.SUCCESS


class TurnToBallNode(BaseNode):

    def __init__(self, node):
        super().__init__(node)
        self.bb = py_trees.blackboard.Client()
        self.bb.register_key(key="/parameters/kp", access=Access.WRITE)
        self.bb.parameters.kp = 0.125

        find_ball = ToBlackboard(name='find_ball',
                                 topic_name="ball_sighting",
                                 topic_type=Sighting,
                                 blackboard_variables={
                                     '/ball_sighting': None
                                 },
                                 qos_profile=10)
        get_turn = TurnToHeading(name='get_turn_value')
        publish_turn = FromBlackboard(name='publish_turn',
                                      topic_name="cmd_vel",
                                      topic_type=Twist,
                                      qos_profile=10,
                                      blackboard_variable="/cmd_vel")

        task = Sequence(name='task', memory=False,
                        children=[find_ball, get_turn, publish_turn])

        self.tree = BehaviourTree(root=task)
        self.tree.setup(node=self.node)

        self.node.create_timer(1, self.tick)

    def tick(self):
        self.log_info('Ticking tree')
        self.tree.tick()


def main():
    rclpy.init()
    node = TurnToBallNode(Node('turn_to_ball'))
    rclpy.spin(node.get_node())


if __name__ == '__main__':
    main()
