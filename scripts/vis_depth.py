#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading # For protecting shared resources

class DepthViewer:
    def __init__(self):
        rospy.init_node('depth_viewer_interactive', anonymous=True)

        # --- Parameters ---
        self.depth_image_topic = rospy.get_param('~depth_image_topic', '/d400/depth/image_rect_raw')
        # Max depth for visualization normalization.
        # For 32FC1 (float, usually meters): this value is in meters.
        # For 16UC1 (uint16, usually mm): this value is in mm (or meters if <100, then auto-multiplied by 1000).
        # Adjust this based on your sensor's typical range for better visualization.
        self.max_depth_viz = rospy.get_param('~max_depth_viz', 5.0) # e.g., 5.0 for meters (32FC1), or 5000 for mm (16UC1)

        # --- Member variables ---
        self.bridge = CvBridge()
        self.cv_depth_image_raw = None    # Raw depth data (e.g., float32 or uint16)
        self.visualization_image = None # Image ready for display (e.g., colormapped uint8)
        self.image_lock = threading.Lock() # To protect access to image data

        self.window_name = "Depth Image Interactive Viewer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # --- ROS Subscriber ---
        self.image_sub = rospy.Subscriber(self.depth_image_topic, Image, self.image_callback)
        
        rospy.loginfo("Depth viewer initialized. Subscribing to: %s", self.depth_image_topic)
        rospy.loginfo("Click on the image to see raw depth pixel values.")
        rospy.on_shutdown(self.cleanup)

    def image_callback(self, ros_image_msg):
        try:
            # Convert ROS Image message to OpenCV image
            # For depth images, common encodings are:
            # - 16UC1: unsigned 16-bit integer
            # - 32FC1: 32-bit float
            cv_image_raw = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # --- 输出深度图类型信息 ---
        rospy.loginfo("=== Depth Image Info ===")
        rospy.loginfo("ROS encoding: %s", ros_image_msg.encoding)
        rospy.loginfo("OpenCV dtype: %s", cv_image_raw.dtype)
        rospy.loginfo("Image shape: %s", cv_image_raw.shape)
        rospy.loginfo("Min value: %s", np.nanmin(cv_image_raw))
        rospy.loginfo("Max value: %s", np.nanmax(cv_image_raw))
        rospy.loginfo("Mean value: %s", np.nanmean(cv_image_raw))
        if ros_image_msg.encoding == "32FC1":
            nan_count = np.count_nonzero(np.isnan(cv_image_raw))
            rospy.loginfo("NaN pixels count: %d", nan_count)
        rospy.loginfo("========================")

        with self.image_lock:
            self.cv_depth_image_raw = cv_image_raw.copy() # Store raw depth data

            # --- Prepare for visualization (this part normalizes for display, not for raw value output) ---
            depth_display = cv_image_raw.copy()

            current_max_depth_viz = self.max_depth_viz
            if ros_image_msg.encoding == "32FC1": # Values are likely in meters
                depth_display_no_nan = np.nan_to_num(depth_display, nan=0.0) # Replace NaN with 0
                # Normalize for visualization: 0 to max_depth_viz (meters) -> 0 to 255
                depth_normalized = np.clip(depth_display_no_nan / current_max_depth_viz, 0.0, 1.0) * 255
            elif ros_image_msg.encoding == "16UC1": # Values are likely in millimeters
                # If max_depth_viz seems to be in meters for a mm stream, convert it
                if current_max_depth_viz > 0 and current_max_depth_viz < 100: # Heuristic: small value likely meters
                    max_depth_mm = current_max_depth_viz * 1000.0
                else:
                    max_depth_mm = current_max_depth_viz # Assume it's already in mm

                # Normalize for visualization: 0 to max_depth_mm -> 0 to 255
                # Ensure max_depth_mm is not zero to avoid division by zero
                if max_depth_mm > 0:
                    depth_normalized = np.clip(depth_display.astype(np.float32) / max_depth_mm, 0.0, 1.0) * 255
                else: # Fallback if max_depth_mm is 0 or invalid
                    depth_normalized = np.zeros_like(depth_display, dtype=np.float32)

            else:
                rospy.logwarn_throttle(5, "Unsupported depth image encoding: %s. Visualizing as is (if possible).", ros_image_msg.encoding)
                if len(cv_image_raw.shape) == 2: # Attempt a simple normalization if it's some other single channel type
                     cv2.normalize(cv_image_raw, depth_display, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = depth_display

            # Convert to 8-bit unsigned integer
            depth_vis_8u = depth_normalized.astype(np.uint8)

            # Apply colormap for better visualization
            self.visualization_image = cv2.applyColorMap(depth_vis_8u, cv2.COLORMAP_JET)
            # For pixels that were 0 (no data or very close), make them black in colormap
            if ros_image_msg.encoding == "32FC1":
                 self.visualization_image[depth_display_no_nan == 0] = [0,0,0]
            elif ros_image_msg.encoding == "16UC1":
                 self.visualization_image[depth_display == 0] = [0,0,0]


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.image_lock:
                if self.cv_depth_image_raw is not None and self.visualization_image is not None:
                    height, width = self.cv_depth_image_raw.shape[:2]
                    if 0 <= y < height and 0 <= x < width:
                        # Get the RAW pixel value from the original depth image
                        depth_value = self.cv_depth_image_raw[y, x]
                        
                        # Create a temporary image for drawing so we don't permanently alter self.visualization_image
                        display_copy = self.visualization_image.copy()
                        
                        # Draw a circle at the clicked point
                        cv2.circle(display_copy, (x, y), 5, (0, 255, 0), -1) # Green circle

                        # Prepare text for the RAW depth value
                        # For float values, format to a few decimal places for readability on screen
                        if isinstance(depth_value, (np.float32, float)):
                            text = "Raw Value: {:.4f}".format(depth_value)
                        else: # For integers (like uint16) or other types
                            text = "Raw Value: {}".format(depth_value)
                        
                        # Log the raw value to the console (full precision for floats)
                        rospy.loginfo("Raw depth value at ({}, {}): {}".format(x, y, depth_value))
                        
                        # Put text on the image
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_pos = (x + 10, y - 10) # Adjust position if near edge
                        # Basic check to prevent text going off-screen
                        (text_width, text_height), _ = cv2.getTextSize(text, font, 0.5, 1)
                        if text_pos[0] + text_width > width: text_pos = (x - text_width - 5, y - 10)
                        if text_pos[1] - text_height < 0: text_pos = (x + 10, y + text_height + 5)
                        if text_pos[0] < 0 : text_pos = (5, text_pos[1]) # Adjust if x goes negative

                        cv2.putText(display_copy, text, text_pos, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        cv2.imshow(self.window_name, display_copy)
                        # cv2.waitKey(1) # Allow display to update, main loop handles continuous display
                    else:
                        rospy.logwarn("Clicked outside image bounds.")
                else:
                    rospy.logwarn("No image data available for mouse click.")

    def run(self):
        rate = rospy.Rate(30) # 30 Hz for display updates
        while not rospy.is_shutdown():
            with self.image_lock:
                if self.visualization_image is not None:
                    cv2.imshow(self.window_name, self.visualization_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC or 'q' key to exit
                rospy.loginfo("Exit requested by user.")
                break
            rate.sleep()
        
        self.cleanup()

    def cleanup(self):
        rospy.loginfo("Shutting down depth viewer.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        viewer = DepthViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logerr("Unhandled exception: %s", str(e))
        # traceback.print_exc() # Uncomment for more detailed exception trace
    finally:
        cv2.destroyAllWindows()