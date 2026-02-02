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
        
        # --- 파라미터 설정 ---
        self.BUBBLE_RADIUS = 0.7        # 장애물 안전 거리
        self.MAX_GAP_DIST = 4.5         # 인지할 최대 거리
        self.STRAIGHT_SPEED = 5.0       # 직선 주로 속도
        self.CORNER_SPEED = 1.0         # 급격한 커브 시 최소 속도 (1.0~1.2 추천)
        
        self.STEER_SMOOTHING = 0.1      # 조향 부드러움 정도
        self.prev_steering_angle = 0.0
        self.MAX_STEER_DIFF = 0.15      # 갑작스러운 조향 변화 제한 (Spin-out 방지)

    def preprocess_lidar(self, ranges, data):
        """
        필터링과 시야 제한을 통해 리다 데이터를 전처리합니다.
        """
        proc_ranges = np.array(ranges)
        proc_ranges[np.isinf(proc_ranges)] = self.MAX_GAP_DIST
        proc_ranges[np.isnan(proc_ranges)] = 0.0

        # 이동 평균 필터 (데이터 평활화)
        kernel = np.ones(5) / 5
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')
        
        # [수정] 시야각을 90도로 제한하여 뒤를 보려는 현상 방지
        angle_min = data.angle_min
        for i in range(len(proc_ranges)):
            angle = angle_min + i * data.angle_increment
            if abs(angle) > math.radians(90): 
                proc_ranges[i] = 0.0
            elif proc_ranges[i] > self.MAX_GAP_DIST:
                proc_ranges[i] = self.MAX_GAP_DIST

        # 급격한 거리 변화(Disparity) 처리
        diff = np.diff(proc_ranges)
        disparities = np.where(abs(diff) > 0.5)[0]
        for idx in disparities:
            start = max(0, idx - 8)
            end = min(len(proc_ranges), idx + 8)
            proc_ranges[start:end] = min(proc_ranges[idx], proc_ranges[idx+1])

        return proc_ranges

    def lidar_callback(self, data):
        proc_ranges = self.preprocess_lidar(data.ranges, data)
        
        valid_indices = np.where(proc_ranges > 0.1)[0]
        if len(valid_indices) == 0: return
        
        closest_idx = valid_indices[np.argmin(proc_ranges[valid_indices])]
        min_dist = proc_ranges[closest_idx]
        
        # 가변 버블 적용: 장애물이 가까우면 길을 찾기 위해 버블을 작게 설정
        dynamic_bubble = self.BUBBLE_RADIUS
        if min_dist < 1.0:
            dynamic_bubble = 0.35 
            
        angle_spread = np.arctan2(dynamic_bubble, min_dist)
        index_spread = int(angle_spread / data.angle_increment)
        
        start_idx = max(0, closest_idx - index_spread)
        end_idx = min(len(proc_ranges) - 1, closest_idx + index_spread)
        proc_ranges[start_idx : end_idx + 1] = 0.0
        
        # 최대 간격(Max Gap) 찾기
        free_space_indices = np.where(proc_ranges > 0.1)[0]
        if len(free_space_indices) == 0:
            self.stop_vehicle()
            return
            
        gaps = np.split(free_space_indices, np.where(np.diff(free_space_indices) > 1)[0] + 1)
        max_gap = max(gaps, key=len)
        best_idx = max_gap[len(max_gap) // 2]
        
        target_angle = (best_idx * data.angle_increment) + data.angle_min
        
        # [핵심 수정 1] 조향각 최대 제한 (40도 이상 꺾지 않게 하여 뱅글 도는 현상 원천 차단)
        MAX_SAFE_STEER = math.radians(40)
        if abs(target_angle) > MAX_SAFE_STEER:
            target_angle = np.sign(target_angle) * MAX_SAFE_STEER
        
        # 조향 변화량 제한
        steer_diff = target_angle - self.prev_steering_angle
        if abs(steer_diff) > self.MAX_STEER_DIFF:
            target_angle = self.prev_steering_angle + np.sign(steer_diff) * self.MAX_STEER_DIFF
        
        # 조향 평활화
        smoothed_angle = (self.STEER_SMOOTHING * target_angle) + ((1 - self.STEER_SMOOTHING) * self.prev_steering_angle)
        self.prev_steering_angle = smoothed_angle
        
        self.publish_drive(smoothed_angle)

    def publish_drive(self, steering_angle):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        
        abs_angle = abs(steering_angle)
        
        # [핵심 수정 2] 단계별 미리 감속 로직
        # 1. 급커브 (20도 이상): 아주 느린 속도로 확실히 회전
        if abs_angle > math.radians(20):
            speed = 1.0
        # 2. 일반 커브 (10도 이상): 적절히 감속
        elif abs_angle > math.radians(10):
            speed = 2.0
        # 3. 완만한 커브 (5도 이상): 가속 준비
        elif abs_angle > math.radians(5):
            speed = 3.5
        # 4. 직선 주로: 최대 속도
        else:
            speed = self.STRAIGHT_SPEED
            
        drive_msg.drive.speed = float(speed)
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
