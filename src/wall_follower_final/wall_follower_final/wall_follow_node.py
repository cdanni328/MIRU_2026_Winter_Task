import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # PID 게인
        self.kp = 3.0  # 너무 높으면 안쪽으로 파고드므로 살짝 하향 (3.5 -> 3.0)
        self.kd = 0.25 # 휘청거림 방지를 위해 D값 유지
        self.prev_error = 0.0

    def get_range(self, range_data, angle):
        index = int((angle - range_data.angle_min) / range_data.angle_increment)
        index = max(0, min(index, len(range_data.ranges) - 1))
        dist = range_data.ranges[index]
        if np.isnan(dist) or np.isinf(dist) or dist > 15.0:
            return 10.0
        return dist

    def get_error(self, range_data, desired_dist):
        angle_b = math.pi / 2          
        angle_a = math.radians(35)     # 조금 더 앞쪽을 보게 수정 (40 -> 35)
        theta = angle_b - angle_a

        dist_b = self.get_range(range_data, angle_b)
        dist_a = self.get_range(range_data, angle_a)

        alpha = math.atan2(dist_a * math.cos(theta) - dist_b, dist_a * math.sin(theta))
        current_distance = dist_b * math.cos(alpha)
        
        look_ahead = 1.2 # 예측 거리를 적절히 조절
        future_distance = current_distance + look_ahead * math.sin(alpha)
        
        return desired_dist - future_distance

    def pid_control(self, error, velocity):
        steering_angle = -(self.kp * error + self.kd * (error - self.prev_error))
        
        # [핵심 수정 1] 좌회전 시 조향각이 너무 급격하면 살짝 완화 (안쪽 긁힘 방지)
        if steering_angle > 0: # 왼쪽으로 꺾는 상황
            steering_angle *= 0.8 # 왼쪽 조향 민감도를 20% 감소시켜 크게 돌게 함
            
        self.prev_error = error

        max_steer = math.radians(25)
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(velocity)
        self.pub_drive.publish(drive_msg)

    def scan_callback(self, msg):
        # [핵심 수정 2] 동적 목표 거리 설정
        # 평소엔 0.8m지만, 코너(error가 클 때)에서는 벽에서 더 떨어지도록 유도
        base_desired_dist = 0.8
        error_for_check = self.get_error(msg, base_desired_dist)
        
        # 왼쪽 코너에 너무 붙는 상황(error가 음수)이면 목표 거리를 일시적으로 늘림
        if error_for_check < -0.1:
            actual_desired_dist = 1.0 # 0.8m에서 1.0m로 목표 상향
        else:
            actual_desired_dist = base_desired_dist
            
        error = self.get_error(msg, actual_desired_dist)
        
        # 속도 조절
        if abs(error) > 0.3:
            velocity = 0.7 
        else:
            velocity = 1.5 
            
        self.pid_control(error, velocity)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

'''
<패치노트>
현재 상황 : 시계방향으로 주행시 아무문제 없이 잘 감
문제점 : 반시계방향으로 주행시 u자 코너를 돌지 못하고 벽에 박음
 지금은 왼쪽 벽만 follow하는 알고리즘을 사용하고 있기 때문에
 u자 코너를 돌 때 멀리있는 벽을 인식해버림
 해결방안 : 왼쪽벽과 오른쪽 벽 중 더 가까운 벽을 인식하도록 설정
 '''