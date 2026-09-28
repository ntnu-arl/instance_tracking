"""ROS message conversions for instance tracking."""

import time
from typing import Optional

import cv_bridge
import numpy as np
import torch
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import ColorRGBA

from instance_tracking_msgs.msg import OpenVocabTrackFeature as OpenVocabTrackFeatureMsg
from instance_tracking_msgs.msg import OpenVocabTrackFeatures as OpenVocabTrackFeaturesMsg
from instance_tracking_msgs.msg import Track as TrackMsg
from instance_tracking_msgs.msg import TrackedInstances as TrackedInstancesMsg
from instance_tracking.tracking.track import Track
from instance_tracking.tracker import TrackerResult


class Conversions:
    """Conversion namespace for ROS messages."""
    
    bridge = cv_bridge.CvBridge()
    
    @classmethod
    def to_image(cls, msg: ImageMsg, encoding: str = "passthrough") -> np.ndarray:
        """Convert sensor_msgs/Image to numpy array.
        
        Args:
            msg: ROS Image message
            encoding: Desired encoding (e.g., "rgb8", "bgr8")
        
        Returns:
            numpy array of the image
        """
        return cls.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
    
    @classmethod
    def to_image_msg(cls, header, img: np.ndarray, encoding: str = "passthrough") -> ImageMsg:
        """Convert numpy array to sensor_msgs/Image.
        
        Args:
            header: ROS header for the message
            img: numpy array image
            encoding: Image encoding
        
        Returns:
            ROS Image message
        """
        msg = cls.bridge.cv2_to_imgmsg(img, encoding=encoding)
        msg.header = header
        return msg
    
    @staticmethod
    def to_track_msg(
        track: Track,
        *,
        include_prototype: bool = True,
        timings: Optional[dict[str, float]] = None,
    ) -> TrackMsg:
        """Convert Track to ROS Track message.
        
        Args:
            track: Track object
        
        Returns:
            ROS Track message
        """
        msg = TrackMsg()
        if track.color is not None:
            msg.has_display_color = True
            r, g, b = track.color
            msg.display_color = ColorRGBA(
                r=float(r) / 255.0,
                g=float(g) / 255.0,
                b=float(b) / 255.0,
                a=1.0,
            )
        else:
            msg.has_display_color = False
        msg.track_id = track.track_id
        msg.instance_id = track.instance_id
        msg.age = track.age
        msg.frames_since_detection = track.frames_since_detection
        msg.confidence = track.confidence
        if include_prototype and track.prototype is not None:
            prototype_start = time.perf_counter()
            msg.prototype = (
                track.prototype.detach().reshape(-1).to(dtype=torch.float32).cpu().tolist()
            )
            if timings is not None:
                timings["tracked_msg_prototype_ms"] = (
                    timings.get("tracked_msg_prototype_ms", 0.0)
                    + (time.perf_counter() - prototype_start) * 1000.0
                )
        return msg
    
    @classmethod
    def to_tracked_instances_msg(
        cls,
        header,
        result: TrackerResult,
        mask_image: Optional[torch.Tensor] = None,
        confidence_image: Optional[torch.Tensor] = None,
        include_confidence: bool = True,
        include_prototypes: bool = True,
        timings: Optional[dict[str, float]] = None,
    ) -> TrackedInstancesMsg:
        """Convert TrackerResult to ROS TrackedInstances message.
        
        Args:
            header: ROS header for the message
            result: TrackerResult from instance tracker
            mask_image: Optional override mask image [H, W] with instance ids
            confidence_image: Optional confidence override aligned to mask_image
        
        Returns:
            ROS TrackedInstances message
        """
        msg = TrackedInstancesMsg()
        msg.header = header
        
        # Convert mask tensor to image message
        mask_start = time.perf_counter()
        mask_tensor = result.masks if mask_image is None else mask_image
        mask_np = mask_tensor.cpu().numpy().astype(np.uint16)
        msg.masks = cls.bridge.cv2_to_imgmsg(mask_np, encoding="mono16")
        msg.masks.header = header
        if timings is not None:
            timings["tracked_msg_mask_ms"] = (
                time.perf_counter() - mask_start
            ) * 1000.0
        
        # Convert tracks
        tracks_start = time.perf_counter()
        msg.tracks = [
            cls.to_track_msg(
                t,
                include_prototype=include_prototypes,
                timings=timings,
            )
            for t in result.tracks
        ]
        if timings is not None:
            timings["tracked_msg_tracks_ms"] = (
                time.perf_counter() - tracks_start
            ) * 1000.0
        
        msg.is_keyframe = result.is_keyframe
        msg.frame_index = result.frame_index

        confidence_tensor = None
        if include_confidence:
            confidence_tensor = (
                result.final_argmax_confidence if confidence_image is None else confidence_image
            )
        if confidence_tensor is not None:
            confidence_start = time.perf_counter()
            conf_np = confidence_tensor.cpu().numpy().astype(np.float32)
            msg.confidence_image = cls.bridge.cv2_to_imgmsg(conf_np, encoding="32FC1")
            msg.confidence_image.header = header
            if timings is not None:
                timings["tracked_msg_confidence_ms"] = (
                    time.perf_counter() - confidence_start
                ) * 1000.0
        elif timings is not None:
            timings["tracked_msg_confidence_ms"] = 0.0

        return msg

    @staticmethod
    def to_open_vocab_track_feature_msg(
        source_track_id: int,
        source_instance_id: int,
        feature: torch.Tensor | np.ndarray,
    ) -> OpenVocabTrackFeatureMsg:
        """Convert one source-track feature into a ROS message."""

        msg = OpenVocabTrackFeatureMsg()
        msg.source_track_id = int(source_track_id)
        msg.source_instance_id = int(source_instance_id)

        if isinstance(feature, torch.Tensor):
            flat = feature.detach().reshape(-1).to(dtype=torch.float32).cpu().tolist()
        else:
            flat = np.asarray(feature, dtype=np.float32).reshape(-1).tolist()

        msg.feature = flat
        return msg

    @classmethod
    def to_open_vocab_track_features_msg(
        cls,
        header,
        frame_index: int,
        encoder_id: str,
        features: list[tuple[int, int, torch.Tensor | np.ndarray]],
    ) -> OpenVocabTrackFeaturesMsg:
        """Convert batched source-track features into a ROS message."""

        msg = OpenVocabTrackFeaturesMsg()
        msg.header = header
        msg.frame_index = int(frame_index)
        msg.encoder_id = encoder_id
        msg.features = [
            cls.to_open_vocab_track_feature_msg(track_id, instance_id, feature)
            for track_id, instance_id, feature in features
        ]
        return msg
