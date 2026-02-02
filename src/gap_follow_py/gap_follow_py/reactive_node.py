import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_node')
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # --- 주행 안정화 파라미터 ---
        self.BUBBLE_RADIUS = 0.8        # 안전 버블
        self.PREPROCESS_CONV_SIZE = 9    # 필터를 더 크게 해서 부드럽게
        self.MAX_GAP_DIST = 10.0          # Tweak 4: 3m 이상은 무시하여 wiggling 방지
        self.STRAIGHT_SPEED = 5.0
        self.CORNER_SPEED = 1.2
        
        # 조향 부드러움 계수 (0.0 ~ 1.0) 
        # 낮을수록 부드럽지만 반응이 느려짐, 0.2~0.4 권장
        self.STEER_SMOOTHING = 0.3 
        self.prev_steering_angle = 0.0

    def preprocess_lidar(self, ranges, data):
        proc_ranges = np.array(ranges)
        
        # 유턴 방지 및 거리 제한 (Tweak 4 반영)
        angle_min = data.angle_min
        for i in range(len(proc_ranges)):
            angle = angle_min + i * data.angle_increment
            if abs(angle) > math.radians(75): # 시야각 150도
                proc_ranges[i] = 0.0
            elif proc_ranges[i] > self.MAX_GAP_DIST:
                proc_ranges[i] = self.MAX_GAP_DIST

        # --- Tweak 2: 간단한 Disparity Extender ---
        # 인접한 레이저 간의 거리 차이가 1m 이상 나면 '절벽'으로 판단
        diff = np.diff(proc_ranges)
        disparities = np.where(abs(diff) > 1.0)[0]
        for idx in disparities:
            # 급격한 변화가 있는 곳 주변 5칸을 강제로 가까운 거리 값으로 고정
            start = max(0, idx - 5)
            end = min(len(proc_ranges), idx + 5)
            proc_ranges[start:end] = min(proc_ranges[idx], proc_ranges[idx+1])

        # 노이즈 제거 필터
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE)/self.PREPROCESS_CONV_SIZE, mode='same')
        return proc_ranges

    def lidar_callback(self, data):
        proc_ranges = self.preprocess_lidar(data.ranges, data)
        
        # 가장 가까운 점 찾기 및 버블 적용
        valid_indices = np.where(proc_ranges > 0.1)[0]
        if len(valid_indices) == 0: return
        
        closest_idx = valid_indices[np.argmin(proc_ranges[valid_indices])]
        min_dist = proc_ranges[closest_idx]
        
        angle_spread = np.arctan2(self.BUBBLE_RADIUS, min_dist)
        index_spread = int(angle_spread / data.angle_increment)
        
        start_idx = max(0, closest_idx - index_spread)
        end_idx = min(len(proc_ranges) - 1, closest_idx + index_spread)
        proc_ranges[start_idx : end_idx + 1] = 0.0
        
        # Max Gap 찾기
        free_space_indices = np.where(proc_ranges > 0.1)[0]
        if len(free_space_indices) == 0:
            self.stop_vehicle()
            return
            
        gaps = np.split(free_space_indices, np.where(np.diff(free_space_indices) > 1)[0] + 1)
        max_gap = max(gaps, key=len)
        
        # Best Point: Gap의 중앙
        best_idx = max_gap[len(max_gap) // 2]
        new_angle = (best_idx * data.angle_increment) + data.angle_min
        
        # --- 핵심: 조향각 평활화 (Smoothing) ---
        # 이전 각도와 새 각도를 섞어서 급격한 변화를 막음
        smoothed_angle = (self.STEER_SMOOTHING * new_angle) + ((1 - self.STEER_SMOOTHING) * self.prev_steering_angle)
        self.prev_steering_angle = smoothed_angle
        
        self.publish_drive(smoothed_angle)

    def publish_drive(self, steering_angle):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        
        # 조향각에 따른 가변 속도
        abs_angle = abs(steering_angle)
        if abs_angle > math.radians(30):
            drive_msg.drive.speed = self.CORNER_SPEED
        else:
            drive_msg.drive.speed = self.STRAIGHT_SPEED
            
        self.pub_drive.publish(drive_msg)

    def stop_vehicle(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        self.pub_drive.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()