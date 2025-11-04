#!/home/jetson-nano/catkin_ws/src/april-course-correction/april_openCV_env/bin/python
"""
ArUco Marker Detection Node
Detects ArUco markers and publishes corner positions to 'april-corners-topic'
"""
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import apriltag
import tf

class AprilCornersPub:
    def __init__(self): # constructor
        rospy.init_node('aruco_corners_pub')

        # april corners publisher
        self.april_corners_pub = rospy.Publisher('aruco_corners_topic', Float32MultiArray, queue_size=10)
        self.pose_pub = rospy.Publisher('april_pose_topic', Float32MultiArray, queue_size=10)

        # zed mini subscriber
        self.RGB_sub = rospy.Subscriber('/zedm/zed_node/left/image_rect_color', Image, self.RGB_callback) 
        rospy.loginfo("Subscribed to ZED Mini RGB feed")

        # instantiates CvBridge object, that converts RGB to format digestible by OpenCV
        self.bridge = CvBridge()

        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

        f = 366 # focal length, UPDATE
        rho = 0.000002 # physical individual square pixel sensor width AND height conversion
        cx = 315
        cy = 178

        self.camera_matrix = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32) # all actually zero, not placeholders, because /image_rect_color is undistorted, I think

        self.tag_size_meters = 0.184 # meters across

        half_size = self.tag_size_meters / 2
        self.object_points = np.array([
            [-half_size, -half_size, 0], # Bottom-left
            [ half_size, -half_size, 0], # Bottom-right
            [ half_size,  half_size, 0], # Top-right
            [-half_size,  half_size, 0]  # Top-left
        ], dtype=np.float32)
        # Can also adjust errorCorrectionRate, minMarkerPerimeterRate, etc.


    def RGB_callback(self, msg):
        # rospy.loginfo("Received an image message")
        # rospy.loginfo(f"Image frame id: {msg.header.frame_id}, height: {msg.height}, width: {msg.width}")
        # if self.depth_image is None:
        #     return # wait until depth frame available

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # rospy.loginfo(f"Converted image shape: {cv_image.shape}, dtype: {cv_image.dtype}")

            results = self.detector.detect(cv_image)
            rospy.loginfo(f"{len(results)} AprilTags detected.")

            image_points = results.corners.astype(np.float32)
            for r in results:
                image_points = r.corners.astype(np.float32)
            # PnP pose estimation
                successBool, rotation, translation = cv2.solvePnP( # successBool just if it worked, rotation and translation w.r.t initial inertial frame
                    self.object_points, 
                    image_points, 
                    self.camera_matrix, 
                    self.dist_coeffs
                )
                if successBool:
                    # --- Publish the Pose as a PoseStamped Message ---
                    
                    # Convert the rotation vector (rvec) to a quaternion
                    rotation_matrix, _ = cv2.Rodrigues(rotation)
                    transform_matrix = np.eye(4)
                    transform_matrix[0:3, 0:3] = rotation_matrix
                    quaternion = tf.transformations.quaternion_from_matrix(transform_matrix)

                    # Create and populate the message
                    pose_msg = PoseStamped()
                    pose_msg.header = msg.header # Use timestamp from the image
                    pose_msg.header.frame_id = "zedm_left_camera_optical_frame"

                    # `tvec` is the translation from camera to tag
                    pose_msg.pose.position.x = translation[0][0]
                    pose_msg.pose.position.y = translation[1][0]
                    pose_msg.pose.position.z = translation[2][0]

                    # `quaternion` is the orientation of the tag
                    pose_msg.pose.orientation.x = quaternion[0]
                    pose_msg.pose.orientation.y = quaternion[1]
                    pose_msg.pose.orientation.z = quaternion[2]
                    pose_msg.pose.orientation.w = quaternion[3]

                    self.pose_pub.publish(pose_msg)

                    # --- (Optional) Visualize the pose for debugging ---
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rotation, translation, 0.1)

                    cv2.imshow("Image", cv_image)
                    # Draw detected corners for visualization
                        # Extract the corners
                    (ptA, ptB, ptC, ptD) = r.corners
                    # Draw the bounding box
                    cv2.line(cv_image, tuple(ptA.astype(int)), tuple(ptB.astype(int)), (0, 255, 0), 2)
                    cv2.line(cv_image, tuple(ptB.astype(int)), tuple(ptC.astype(int)), (0, 255, 0), 2)
                    cv2.line(cv_image, tuple(ptC.astype(int)), tuple(ptD.astype(int)), (0, 255, 0), 2)
                    cv2.line(cv_image, tuple(ptD.astype(int)), tuple(ptA.astype(int)), (0, 255, 0), 2)
                    # Draw the center
                    (cX, cY) = (int(r.center[0]), int(r.center[1]))
                    cv2.circle(cv_image, (cX, cY), 5, (0, 0, 255), -1)


                cv2.imshow("Image", cv_image)
                cv2.waitKey(1)

            else:
                rospy.logerr(f"An error occurred in RGB_callback")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        print("OpenCV version:", cv2.__version__)
        node = AprilCornersPub()
        node.run()
    except rospy.ROSInterruptException:
        pass