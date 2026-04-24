#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.action import ActionServer
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import torch
import yaml 
from interfaces import valvecoord
class ValveDetectionNode(Node):
    def __init__(self, name):
        super().__init__(name)
        ENGINE_MODEL_PATH = "/home/jetson/ultralytics_robot/best.engine"
        self.model = YOLO(ENGINE_MODEL_PATH, task="detect")
        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info = None
        self.img_size = 640  
        self.names = self.model.names  # 获取类别名称列表
        self.valve_pub = self.create_publisher(
            valvecoord,
            '/valve/position',
            10
        )
        
        self.get_logger().info('Valve Detection Node has been started.')

        self.rgb_subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            1  # 队列大小1：只保留最新帧
            )  
        
        self.depth_subscription = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            queue_size=1) 

    def depth_callback(self, msg):
        try:
        # 将ROS消息转换为OpenCV格式（passthrough保持原始编码，通常为16位深度值，单位毫米）
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            # 每5秒最多打印一次警告（避免日志刷屏）
            self.get_logger().logwarn_throttle(5, f"深度图解析失败: {e}")
        self.depth_image = depth_image

    def image_callback(self, msg):
        if self.depth_image is None or self.camera_info is None:
            self.get_logger().logwarn_throttle(5, "等待深度图和相机内参加载完成...")
            return
        
        try: 
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # 转换为BGR格式
            results = self.model(
            frame,           # 直接用原始 BGR 图像
            imgsz=self.img_size,
            conf=0.65,
            iou=0.45,
            verbose=False,
            )
        # 将 Ultralytics 输出转换为和原来一样结构的 dets（列表，每个元素是 [x1,y1,x2,y2,conf,cls] 的 tensor）
            if not results:
                # 没有结果时，构造一个空 det，后面的 for det in dets 逻辑保持一致
                dets = [torch.empty((0, 6))]
            else:  
                res = results[0]
                boxes = res.boxes  # Boxes 对象

                if boxes is None or len(boxes) == 0:
                    dets = [torch.empty((0, 6))]
                else:
                    # 拼成和原来 det 一样的格式：[x1,y1,x2,y2,conf,cls]
                    det = torch.cat(
                        (
                            boxes.xyxy,                   # (N,4)
                            boxes.conf.view(-1, 1),       # (N,1)
                            boxes.cls.view(-1, 1),        # (N,1)
                        ),
                        dim=1,
                    )
                    dets = [det]

            for det in dets:
                if det is None or len(det) == 0:
                    self.get_logger().info(f"无检测结果")
                    continue  # 无检测结果，跳过

                
                img_center_x = frame.shape[1] / 2
                img_center_y = frame.shape[0] / 2
                famen_det = det[det[:, 5] == 0]
                small_det = det[det[:, 5] == 1]

                

                xyz, f_cls, x1, x2, y1, y2 = None, None, None, None, None, None
                xyz_small, s_cls = None, None

                if len(famen_det) > 0:
                    max_conf_idx_famen = torch.argmax(famen_det[:, 4]).item()
                    
                    xyz, f_cls, x1, x2, y1, y2 = self.get_xyz(famen_det, max_conf_idx_famen, frame)
                else:
                    self.get_logger().info(f"未检测到阀门主体")

                if len(small_det) > 0:
                    max_conf_idx_small = torch.argmax(small_det[:, 4]).item()
                    xyz_small, s_cls = self.get_xyz(small_det, max_conf_idx_small, frame)
                else:
                    self.get_logger().info(f"未检测到小目标")
                #这里的xyz和xyz_small的格式是[x,y,z]的列表
                msg = valvecoord()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "camera_color_optical_frame"

                if xyz is not None:
                    msg.x = float(xyz[0])
                    msg.y = float(xyz[1])
                    msg.z = float(xyz[2])
                    msg.valid = True
                    msg.is_small = False
                    self.valve_pub.publish(msg)

                if xyz_small is not None:
                    msg = valvecoord()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "camera_color_optical_frame"
                    msg.x = float(xyz_small[0])
                    msg.y = float(xyz_small[1])
                    msg.z = float(xyz_small[2])
                    msg.valid = True
                    msg.is_small = True
                    self.valve_pub.publish(msg)
                    
            try:
                cv2.imshow("YOLOv11 Detection", frame)
                # 等待1ms按键输入（不阻塞主线程）
                cv2.waitKey(1)
                
            except Exception as e:
                self.get_logger().error(f"显示失败: {e}")
        except Exception as e:
            self.get_logger().error(f"图像处理出错: {e}")
        
        
    def pixel_to_3d(self, u, v, depth_img, cam_info_dict):
        """
        根据像素坐标和深度值，计算目标在相机坐标系下的3D坐标
        原理：针孔相机模型，通过内参将像素坐标反投影到3D空间
        参数：
            u, v: 目标像素坐标（图像平面）
            depth_img: 深度图像（OpenCV格式，值为深度，单位毫米）
            cam_info_dict: 相机内参字典（包含焦距、主点等）
        返回：
            3D坐标 tuple (X, Y, Z)（单位：米），若无效则返回None
        """
        # 从内参字典中提取焦距(fx, fy)和主点坐标(cx, cy)
        fx = cam_info_dict['camera_matrix']['data'][0]  # 焦距x
        fy = cam_info_dict['camera_matrix']['data'][4]  # 焦距y
        cx = cam_info_dict['camera_matrix']['data'][2]  # 主点x
        cy = cam_info_dict['camera_matrix']['data'][5]  # 主点y

        # 检查像素坐标是否在图像范围内（防止越界）
        if v >= depth_img.shape[0] or u >= depth_img.shape[1] or v < 0 or u < 0:
            return None

        # 获取深度值（单位：毫米），若为0则表示深度无效
        d = depth_img[int(v), int(u)]
        if d == 0:
            return None  # 深度无效，返回None
        d = d / 1000.0  # 转换为米

        # 计算相机坐标系下的3D坐标（右手坐标系：X右，Y下，Z前）
        X = (u - cx) * d / fx  # X坐标
        Y = (v - cy) * d / fy  # Y坐标
        Z = d                  # Z坐标（深度）
        return (X, Y, Z)
    
    def load_camera_info(self, yaml_file):
        """
        从YAML文件加载相机内参（焦距、主点等）
        参数：
            yaml_file: 相机内参YAML文件路径
        返回：
            包含相机内参的字典（如camera_matrix、distortion_coefficients等）
        """
        with open(yaml_file, 'r') as file:
            cam_info = yaml.safe_load(file)  # 安全加载YAML内容
        self.get_logger().info(f"加载的相机内参: {cam_info}")  # 打印内参信息（调试用）
        self.camera_info = cam_info  

    def get_xyz(self, det, max_conf_idx, frame):
        x1, y1, x2, y2, conf, cls = det[max_conf_idx].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # 转换为整数坐标
        cls = int(cls)  # 类别索引

        # 在图像上绘制检测框和标签（可视化）
        label = f'{self.names[cls]} {conf:.2f}'  # 标签：类别+置信度
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 标签文本

        # 计算目标中心点像素坐标
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        # 计算目标在相机坐标系下的3D坐标
        xyz = self.pixel_to_3d(x_center, y_center, self.depth_image, self.camera_info)
        if cls==0:
            return xyz,cls,x1,x2,y1,y2
        elif cls==1:
            return xyz,cls    

def main(args=None):
    rclpy.init(args=args)
    valve_detection_node = ValveDetectionNode('valve_detection_node')
    valve_detection_node.load_camera_info('/home/jetson/ultralytics_robot/src/robot_valve_control/robot_valve_control/camera_info.yaml')
    rclpy.spin(valve_detection_node)
    valve_detection_node.destroy_node()
    rclpy.shutdown()