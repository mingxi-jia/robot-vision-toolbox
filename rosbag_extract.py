import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import os
import json
import numpy as np

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        self.bridge = CvBridge()
        self.color_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.color_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/rgb/camera_info',
            self.caminfo_callback,
            10
        )

        self.output_dir = "/home/xhe71/Desktop/robotool_data/"
        os.makedirs(os.path.join(self.output_dir, "Color"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "Depth"), exist_ok=True)

        self.color_count = 0
        self.depth_count = 0
        self.intrinsics_saved = False

    def color_callback(self, msg):
        self.get_logger().info(f"Color image encoding: {msg.encoding}")

        # Convert ROS Image message to numpy array directly
        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)  # bgra8 has 4 channels
        img_np = img_np[:, :, :3]  # Drop alpha

        filename = os.path.join(self.output_dir, "Color", f"color_{self.color_count:06d}.png")
        img_np = img_np.astype(np.uint8)

        cv2.imwrite(filename, img_np)
        self.get_logger().info(f"Saved {filename}")
        self.color_count += 1
        cv2.imshow("Color Image", img_np)
        cv2.waitKey(1)



    def depth_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}")
            return

        # Save .npy (full-precision depth in meters)
        filename_npy = os.path.join(self.output_dir, "Depth", f"depth_{self.depth_count:06d}.npy")
        np.save(filename_npy, img)

        # Save .png for visualization only (scaled to 8-bit)
        filename = os.path.join(self.output_dir, "Depth", f"depth_{self.depth_count:06d}.png")
        img = (img * 1000).astype(np.uint16)
        cv2.imwrite(filename, img)
        self.get_logger().info(f"Saved {filename}")
        self.depth_count += 1
        cv2.imshow("Depth Image", img)
        cv2.waitKey(1)
        
        # self.get_logger().info(f"Saved {filename_npy} and {filename_png}")
        # self.depth_count += 1
        # cv2.imshow("Depth Image", depth_vis)
        # cv2.waitKey(1)


    def caminfo_callback(self, msg):
        if not self.intrinsics_saved:
            intrinsics = {
                "width": msg.width,
                "height": msg.height,
                "fx": msg.k[0],
                "fy": msg.k[4],
                "cx": msg.k[2],
                "cy": msg.k[5],
                "distortion_model": msg.distortion_model,
                "distortion_coefficients": list(msg.d)
            }
            with open(os.path.join(self.output_dir, "camera_intrinsics.json"), 'w') as f:
                json.dump(intrinsics, f, indent=4)
            self.get_logger().info("Saved camera intrinsics.")
            self.intrinsics_saved = True

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
