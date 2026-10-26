#!/usr/bin/env python3
"""Student template for the holonomic obstacle wall follower."""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


def clamp(value, low, high):
    return max(low, min(high, value))


class HolonomicObstacleWallFollower(Node):
    """Complete the LiDAR processing, state machine and control laws."""

    def __init__(self):
        super().__init__('holonomic_obstacle_wall_follower')

        # Desired distance between the robot and the right wall (m).
        self.follow_distance = float(
            self.declare_parameter('follow_distance', 0.35).value
        )
        # Maximum lateral distance used to classify a short obstacle (m).
        self.max_front_length = float(
            self.declare_parameter('max_front_length', 0.50).value
        )
        # Forward velocity (m/s).
        self.forward_speed = float(
            self.declare_parameter('forward_speed', 0.20).value
        )
        # Maximum lateral velocity (m/s).
        self.lateral_speed = float(
            self.declare_parameter('lateral_speed', 0.20).value
        )
        # Turn speed and maximum angular correction (rad/s).
        self.turn_speed = float(
            self.declare_parameter('turn_speed', 1.00).value
        )
        # Maximum LiDAR distance considered (m).
        self.detection_limit = float(
            self.declare_parameter('detection_limit', 1.20).value
        )
        # Proportional gain for lateral-distance correction.
        self.lateral_gain = float(
            self.declare_parameter('lateral_gain', 2.0).value
        )
        # Proportional gain for angular alignment.
        self.alignment_gain = float(
            self.declare_parameter('alignment_gain', 2.0).value
        )

        self.RIGHT = 'RIGHT'
        self.FRONT = 'FRONT'
        self.FRONT_RIGHT = 'FRONT_RIGHT'
        self.BACK_RIGHT = 'BACK_RIGHT'

        self.diagonal_factor = 0.5
        self.region_confirmations = 3
        self.robust_rays = 3

        self.state = self.RIGHT
        self.obstacle_is_long = False
        self.passing_short_obstacle = False
        self.command = Twist()
        self.odom_pose = None
        self.front_start_pose = None
        self.right_angle = None

        self.region_candidate = None
        self.region_candidate_count = 0
        self.stable_region = None

        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.publish_command)

        self.get_logger().info(
            'Holonomic obstacle wall follower started: state RIGHT'
        )

    def publish_command(self):
        self.publisher.publish(self.command)

    def odom_callback(self, msg):
        """Store planar pose (x, y, yaw) from odometry."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        quaternion = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        _, _, yaw = euler_from_quaternion(quaternion)
        self.odom_pose = (position.x, position.y, yaw)

    def region_measurements(self, scan):
        """Return robust distances and angles for the four LiDAR regions."""
        # TODO 1: Create FRONT, FRONT_RIGHT, RIGHT and BACK_RIGHT lists.
        # Store each valid ray as (distance, angle_deg).
        regions = {
            self.FRONT: [],
            self.FRONT_RIGHT: [],
            self.RIGHT: [],
            self.BACK_RIGHT: [],
        }
        for i, distance in enumerate(scan.ranges):
            # Ignore invalid (inf/nan) and out-of-range measurements.
            if not math.isfinite(distance):
                continue
            if not scan.range_min < distance < scan.range_max:
                continue
            # Ray angle in degrees, normalised to [-180, 180).
            angle = math.degrees(scan.angle_min + i * scan.angle_increment)
            angle = (angle + 180.0) % 360.0 - 180.0

            if -20.0 <= angle <= 20.0:
                regions[self.FRONT].append((distance, angle))
            elif -70.0 <= angle < -20.0:
                regions[self.FRONT_RIGHT].append((distance, angle))
            elif -110.0 <= angle < -70.0:
                regions[self.RIGHT].append((distance, angle))
            elif -160.0 <= angle < -110.0:
                regions[self.BACK_RIGHT].append((distance, angle))

        # TODO 2: For every region, select the three nearest valid rays and
        # return their mean distance and mean angle. Use math.inf and None
        # when a region has no valid ray.
        measurements = {}
        for region, rays in regions.items():
            if not rays:
                measurements[region] = (math.inf, None)
                continue
            nearest = sorted(rays)[:self.robust_rays]
            mean_distance = sum(d for d, _ in nearest) / len(nearest)
            mean_angle = sum(a for _, a in nearest) / len(nearest)
            measurements[region] = (mean_distance, mean_angle)
        return measurements

    def closest_region(self, measurements):
        """Return the closest region inside detection_limit."""
        # TODO 3: Ignore regions whose distance is not below detection_limit.
        visible = {
            region: distance
            for region, (distance, _) in measurements.items()
            if distance < self.detection_limit
        }
        if not visible:
            return None
        return min(visible, key=visible.get)

    def confirmed_region(self, measurements):
        """Confirm a region change over several consecutive scans."""
        # TODO 4: Update region_candidate, region_candidate_count and
        # stable_region using self.region_confirmations.
        closest = self.closest_region(measurements)
        if closest == self.stable_region:
            # No change: forget any half-confirmed candidate.
            self.region_candidate = None
            self.region_candidate_count = 0
        elif closest == self.region_candidate:
            self.region_candidate_count += 1
        else:
            self.region_candidate = closest
            self.region_candidate_count = 1

        if self.region_candidate_count >= self.region_confirmations:
            self.stable_region = self.region_candidate
            self.region_candidate = None
            self.region_candidate_count = 0
        return self.stable_region

    def set_state(self, state):
        """Change state and log the transition."""
        if state != self.state:
            self.get_logger().info(f'STATE {self.state} -> {state}')
            self.state = state

    def lateral_progress(self):
        """Return travel along the robot's initial lateral axis."""
        # TODO 5: Project odometry displacement using:
        # abs(-sin(yaw0) * dx + cos(yaw0) * dy)
        if self.odom_pose is None or self.front_start_pose is None:
            return 0.0
        x0, y0, yaw0 = self.front_start_pose
        x, y, _ = self.odom_pose
        dx = x - x0
        dy = y - y0
        return abs(-math.sin(yaw0) * dx + math.cos(yaw0) * dy)

    def start_front(self):
        """Enter FRONT for a new frontal obstacle."""
        # Save the entry pose to measure lateral progress in FRONT.
        self.front_start_pose = self.odom_pose
        self.obstacle_is_long = False
        self.passing_short_obstacle = False
        self.set_state(self.FRONT)

    def follow_right(self, measurements, closest, cmd):
        """Follow the wall using lateral and angular correction."""
        # TODO 6:
        # - Change to FRONT when the closest region is FRONT.
        # - Correct right distance with linear.y.
        # - Move the closest RIGHT angle towards -90 degrees.
        # - While passing_short_obstacle is True, keep angular.z at zero.
        if closest == self.FRONT:
            if self.odom_pose is not None:
                self.start_front()
            return
        if closest == self.BACK_RIGHT and self.passing_short_obstacle:
            # The obstacle corner is now behind: finish the bypass.
            self.set_state(self.BACK_RIGHT)
            return

        distance, angle = measurements[self.RIGHT]
        if angle is None:
            return  # No right wall visible: stay stopped.
        self.right_angle = angle

        cmd.linear.x = self.forward_speed
        # Positive linear.y moves left, away from a too-close right wall.
        cmd.linear.y = clamp(
            self.lateral_gain * (self.follow_distance - distance),
            -self.lateral_speed, self.lateral_speed,
        )
        if self.passing_short_obstacle:
            # The closest point is the obstacle corner, not a wall.
            cmd.angular.z = 0.0
        else:
            # Angle above -90 deg means the nose points into the wall:
            # positive error -> turn left.
            angle_error = math.radians(angle + 90.0)
            cmd.angular.z = clamp(
                self.alignment_gain * angle_error,
                -self.turn_speed, self.turn_speed,
            )

    def follow_front(self, measurements, closest, cmd):
        """Move laterally and classify the obstacle as short or long."""
        # TODO 7: Move with +vy, measure lateral progress, and change state.
        if self.obstacle_is_long:
            # Long wall: turn left until it leaves FRONT.
            if closest in (self.FRONT_RIGHT, self.RIGHT):
                self.set_state(self.FRONT_RIGHT)
                return
            cmd.angular.z = self.turn_speed
            return

        if closest != self.FRONT:
            # FRONT cleared before max_front_length: short obstacle.
            self.get_logger().info(
                f'Short obstacle (progress {self.lateral_progress():.2f} m)'
            )
            self.passing_short_obstacle = True
            self.set_state(self.FRONT_RIGHT)
            return

        if self.odom_pose is None:
            return
        if self.lateral_progress() >= self.max_front_length:
            self.get_logger().info('Long wall: turning left')
            self.obstacle_is_long = True
            cmd.angular.z = self.turn_speed
            return

        # Stop forward motion and move left.
        cmd.linear.y = self.lateral_speed

    def follow_front_right(self, measurements, closest, cmd):
        """Move diagonally around the front-right side."""
        # TODO 8: Handle short-obstacle and long-wall transitions.
        if self.obstacle_is_long:
            if closest == self.RIGHT:
                # The long wall is now the right wall.
                self.obstacle_is_long = False
                self.set_state(self.RIGHT)
                return
            cmd.angular.z = self.turn_speed
            return

        if closest == self.FRONT:
            if self.odom_pose is not None:
                self.start_front()
            return
        if closest == self.RIGHT:
            # Temporary RIGHT phase: passing_short_obstacle keeps w = 0.
            self.set_state(self.RIGHT)
            return

        # Move forward and keep follow_distance from the obstacle corner.
        cmd.linear.x = self.diagonal_factor * self.forward_speed
        distance, _ = measurements[self.FRONT_RIGHT]
        if distance < self.detection_limit:
            cmd.linear.y = clamp(
                self.lateral_gain * (self.follow_distance - distance),
                -self.lateral_speed, self.lateral_speed,
            )

    def follow_back_right(self, measurements, closest, cmd):
        """Complete the bypass and recover the right wall."""
        # TODO 9: Move diagonally and clear passing_short_obstacle when RIGHT
        # is recovered.
        if closest == self.FRONT:
            if self.odom_pose is not None:
                self.start_front()
            return
        if closest == self.RIGHT:
            # Back on a wall: enable angular alignment again.
            self.passing_short_obstacle = False
            self.set_state(self.RIGHT)
            return

        # Move forward-right around the corner, keeping follow_distance.
        cmd.linear.x = self.diagonal_factor * self.forward_speed
        distance, _ = measurements[self.BACK_RIGHT]
        if distance < self.detection_limit:
            cmd.linear.y = clamp(
                self.lateral_gain * (self.follow_distance - distance),
                -self.lateral_speed, self.lateral_speed,
            )
        else:
            cmd.linear.y = -self.diagonal_factor * self.lateral_speed

    def scan_callback(self, scan):
        """Read the LiDAR and calculate the next velocity command."""
        # TODO 10: Calculate measurements, confirm the closest region, create
        # a new Twist and call the method associated with self.state.
        measurements = self.region_measurements(scan)
        closest = self.confirmed_region(measurements)

        cmd = Twist()
        handlers = {
            self.RIGHT: self.follow_right,
            self.FRONT: self.follow_front,
            self.FRONT_RIGHT: self.follow_front_right,
            self.BACK_RIGHT: self.follow_back_right,
        }
        handlers[self.state](measurements, closest, cmd)
        self.command = cmd

        right_distance, right_angle = measurements[self.RIGHT]
        right_text = (
            '---' if right_angle is None
            else f'{right_distance:.2f} m @ {right_angle:.0f} deg'
        )
        self.get_logger().info(
            f'state={self.state} closest={closest} RIGHT={right_text} '
            f'-> vx={cmd.linear.x:.2f} vy={cmd.linear.y:.2f} '
            f'w={cmd.angular.z:.2f}',
            throttle_duration_sec=0.5,
        )

    def stop(self):
        """Publish a final zero command."""
        self.command = Twist()
        self.timer.cancel()
        try:
            self.publisher.publish(self.command)
        except Exception:
            pass


def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = HolonomicObstacleWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

