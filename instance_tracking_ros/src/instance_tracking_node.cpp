#include <memory>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "instance_tracking/tracker.h"

namespace instance_tracking {

class InstanceTrackingNode : public rclcpp::Node {
 public:
  explicit InstanceTrackingNode(const rclcpp::NodeOptions& options)
      : Node("instance_tracking_node", options) {
    RCLCPP_INFO(get_logger(), "InstanceTrackingNode initialized");

    // Initialize tracker
    tracker_ = std::make_unique<Tracker>();

    // Subscribe to RGB image
    image_sub_ = image_transport::create_subscription(
        this,
        "image",
        [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
          imageCallback(msg);
        },
        "raw");

    // Publisher for segmentation output
    label_pub_ = image_transport::create_publisher(this, "labels/image_raw");
  }

 private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    // TODO: Implement tracking pipeline
    // 1. Convert ROS image to OpenCV
    // 2. Run tracker (sparse FastSAM + DINOv3 propagation)
    // 3. Publish label image
    (void)msg;  // Suppress unused warning for now
    RCLCPP_DEBUG(get_logger(), "Received image");
  }

  std::unique_ptr<Tracker> tracker_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher label_pub_;
};

}  // namespace instance_tracking

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(instance_tracking::InstanceTrackingNode)
