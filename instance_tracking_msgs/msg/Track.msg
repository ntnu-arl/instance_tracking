# Represents a tracked instance

std_msgs/ColorRGBA display_color         # Visualization color (stable within track lifetime)
bool has_display_color                   # Whether display_color is valid

uint32 track_id                    # Persistent track ID (unique across sequence)
uint32 instance_id                 # Pixel value in mask image (1-indexed)
uint32 age                         # Frames since track creation
uint32 frames_since_detection      # Frames since last keyframe detection
float32 confidence                 # Track confidence score [0, 1]
float32[] prototype                # Optional DINO prototype (empty if unavailable)
