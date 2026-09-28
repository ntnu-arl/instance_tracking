# Message containing tracked instance masks and metadata

std_msgs/Header header

# Instance mask image where pixel value = instance_id (0 = background)
sensor_msgs/Image masks

# Track metadata for each active instance
instance_tracking_msgs/Track[] tracks

# Whether masks are from instance segmenter (keyframe) or propagated
bool is_keyframe

# Frame index in current sequence
uint32 frame_index

# Per-pixel argmax confidence of the winning track [H, W], 32FC1, values in [0, 1].
# 0 indicates background or unknown. Empty if not available.
sensor_msgs/Image confidence_image
