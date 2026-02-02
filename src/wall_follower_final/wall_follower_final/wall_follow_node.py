import rclpy
from rclpy.node import Node
import numpy as np
import math # 추가됨
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.sub_scan = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # 게인값 (주행하며 튜닝이 필요할 수 있습니다)
        self.kp = 1.0 
        self.kd = 0.05 
        self.ki = 0.0 

        self.prev_error = 0.0
        self.integral = 0.0

    def get_range(self, range_data, angle):
        index = int((angle - range_data.angle_min) / range_data.angle_increment)
        index = max(0, min(index, len(range_data.ranges) - 1))
        dist = range_data.ranges[index]
        if np.isnan(dist) or np.isinf(dist):
            return 4.0
        return dist

    def get_error(self, range_data, dist):
        angle_b = math.pi / 2          
        angle_a = math.pi / 4          
        theta = angle_b - angle_a      

        dist_b = self.get_range(range_data, angle_b)
        dist_a = self.get_range(range_data, angle_a)

        alpha = math.atan2(dist_a * math.cos(theta) - dist_b, dist_a * math.sin(theta))
        current_distance = dist_b * math.cos(alpha)
        
        look_ahead = 0.8 # L 값
        future_distance = current_distance + look_ahead * math.sin(alpha)
        return dist - future_distance

    def pid_control(self, error, velocity):
        # /drive를 실제로 반환(발행)하는 핵심 부분
        steering_angle = -(self.kp * error + self.kd * (error - self.prev_error))
        self.prev_error = error

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(velocity)
        self.pub_drive.publish(drive_msg)

    def scan_callback(self, msg):
        desired_dist = 0.8 
        error = self.get_error(msg, desired_dist)
        
        # 속도 결정 로직
        if abs(error) > 0.3:
            velocity = 0.5
        else:
            velocity = 1.2 # 안전을 위해 약간 낮춤
            
        self.pid_control(error, velocity)

def main(args=None):
    # ROS 2 파이썬 클라이언트 라이브러리 초기화
    rclpy.init(args=args)
    
    # 우리가 만든 WallFollow 클래스를 인스턴스로 생성
    wall_follow_node = WallFollow()
    
    # 노드가 종료될 때까지 계속 실행 (데이터 수신 및 콜백 실행)
    rclpy.spin(wall_follow_node)
    
    # 종료 시 노드 파괴 및 라이브러리 종료
    wall_follow_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()