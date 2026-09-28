"""Queue-based image processor for ROS2."""

import queue
import threading
import time
from dataclasses import dataclass

import rclpy
import sensor_msgs.msg
from spark_config import Config

from instance_tracking_ros.cadence import should_process_camera_frame
from instance_tracking_ros.ros_conversions import Conversions


@dataclass
class ImageWorkerConfig(Config):
    """Configuration for image worker."""
    
    encoding: str = "rgb8"
    queue_size: int = 1
    subscription_queue_size: int = 1
    save_cache_queue_size: int = 32
    save_cache_subscription_queue_size: int = 32
    drop_warn_every: int = 25
    min_separation_s: float = 0.0
    process_stride: int = 1
    
    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@dataclass
class ImageWorkerFrame:
    """A selected RGB frame plus live-input timing metadata."""

    msg: sensor_msgs.msg.Image
    camera_frame_index: int
    receive_wall_time_s: float
    enqueue_wall_time_s: float
    process_start_wall_time_s: float = 0.0


class ImageWorker:
    """Class to simplify asynchronous message processing.
    
    Runs a background thread that processes images from a queue,
    allowing the ROS callback to return immediately.
    """
    
    def __init__(
        self,
        node,
        config: ImageWorkerConfig,
        topic: str,
        callback,
        *,
        keyframe_interval: int = 0,
        status_log_interval_s: float = 0.0,
        log_stride_skips: bool = False,
    ):
        """Register worker with ROS.
        
        Args:
            node: ROS2 node
            config: ImageWorkerConfig
            topic: Topic to subscribe to
            callback: Function to call with (header, img) for each frame
        """
        self._node = node
        self._node.context.on_shutdown(self.stop)
        
        self._config = config
        self._callback = callback
        self._keyframe_interval = int(keyframe_interval)
        self._status_log_interval_s = max(0.0, float(status_log_interval_s))
        self._log_stride_skips = bool(log_stride_skips)
        self._last_status_log_wall_time_s = time.perf_counter()
        
        self._started = False
        self._should_shutdown = False
        self._last_stamp = None
        self._next_camera_frame_index = 0
        self._received_messages = 0
        self._enqueued_messages = 0
        self._processed_messages = 0
        self._stride_skips = 0
        self._queue_full_drops = 0
        self._min_separation_skips = 0
        self._callback_failures = 0
        
        self._queue = queue.Queue(maxsize=config.queue_size)
        self._sub = node.create_subscription(
            sensor_msgs.msg.Image,
            topic,
            self.add_message,
            config.subscription_queue_size,
        )
        self.start()
    
    def add_message(self, msg):
        """Add new message to queue (called from ROS callback)."""
        receive_wall_time_s = time.perf_counter()
        camera_frame_index = self._next_camera_frame_index
        self._next_camera_frame_index += 1
        self._received_messages += 1
        if not should_process_camera_frame(
            camera_frame_index,
            process_stride=self._config.process_stride,
            keyframe_interval=self._keyframe_interval,
        ):
            self._stride_skips += 1
            self._log_stride_skip_warning(msg, camera_frame_index)
            self._maybe_log_status(receive_wall_time_s)
            return

        frame = ImageWorkerFrame(
            msg=msg,
            camera_frame_index=camera_frame_index,
            receive_wall_time_s=receive_wall_time_s,
            enqueue_wall_time_s=time.perf_counter(),
        )
        if not self._queue.full():
            try:
                self._queue.put(frame, block=False, timeout=False)
                self._enqueued_messages += 1
            except queue.Full:
                self._queue_full_drops += 1
                self._log_drop_warning(msg, camera_frame_index, race=True)
        else:
            self._queue_full_drops += 1
            self._log_drop_warning(msg, camera_frame_index, race=False)
        self._maybe_log_status(receive_wall_time_s)
    
    def start(self):
        """Start worker processing queue."""
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._do_work)
            self._thread.start()
    
    def stop(self):
        """Stop worker from processing queue."""
        if self._started:
            self._should_shutdown = True
            self._thread.join()
        
        self._started = False
        self._should_shutdown = False
    
    def spin(self):
        """Wait for ROS to shutdown or worker to exit."""
        if not self._started:
            return
        
        while self._thread.is_alive() and not self._should_shutdown:
            time.sleep(1.0e-2)
        
        self.stop()
    
    def _do_work(self):
        """Worker thread main loop."""
        while not self._should_shutdown:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            frame.process_start_wall_time_s = time.perf_counter()
            msg = frame.msg
            
            # Rate limiting based on timestamp
            curr_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            if self._last_stamp is not None:
                diff_s = 1.0e-9 * (curr_stamp - self._last_stamp).nanoseconds
                if diff_s < self._config.min_separation_s:
                    self._min_separation_skips += 1
                    continue
            
            self._last_stamp = curr_stamp
            
            try:
                img = Conversions.to_image(msg, encoding=self._config.encoding)
                self._callback(msg.header, img, frame)
                self._processed_messages += 1
            except Exception as e:
                import traceback
                self._callback_failures += 1
                self._node.get_logger().error(f"Image processing failed: {e}\n{traceback.format_exc()}")

    def stats(self) -> dict:
        """Return current image-worker counters for monitoring and shutdown logs."""
        return {
            "received": self._received_messages,
            "enqueued": self._enqueued_messages,
            "processed": self._processed_messages,
            "stride_skips": self._stride_skips,
            "queue_full_drops": self._queue_full_drops,
            "min_separation_skips": self._min_separation_skips,
            "callback_failures": self._callback_failures,
            "queue_depth": self._queue.qsize(),
            "queue_size": self._config.queue_size,
            "subscription_queue_size": self._config.subscription_queue_size,
            "process_stride": self._config.process_stride,
            "keyframe_interval": self._keyframe_interval,
        }

    def _log_drop_warning(self, msg, camera_frame_index: int, *, race: bool) -> None:
        if self._config.drop_warn_every <= 0:
            return
        if self._queue_full_drops == 1 or self._queue_full_drops % self._config.drop_warn_every == 0:
            mode = "queue full race" if race else "queue full"
            self._node.get_logger().warn(
                "Dropped "
                f"{self._queue_full_drops} input frames in ImageWorker "
                f"({mode}, queue={self._config.queue_size}, "
                f"sub_queue={self._config.subscription_queue_size}, "
                f"received={self._received_messages}, "
                f"enqueued={self._enqueued_messages}, "
                f"processed={self._processed_messages}, "
                f"stride_skips={self._stride_skips}, "
                f"camera_frame_index={camera_frame_index}, "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d})"
            )

    def _log_stride_skip_warning(self, msg, camera_frame_index: int) -> None:
        if not self._log_stride_skips or self._config.drop_warn_every <= 0:
            return
        if self._stride_skips == 1 or self._stride_skips % self._config.drop_warn_every == 0:
            self._node.get_logger().info(
                "Skipped "
                f"{self._stride_skips} input frames by process_stride "
                f"(stride={self._config.process_stride}, "
                f"keyframe_interval={self._keyframe_interval}, "
                f"camera_frame_index={camera_frame_index}, "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d})"
            )

    def _maybe_log_status(self, now_wall_time_s: float) -> None:
        """Emit a sparse health summary while input is flowing."""

        if self._status_log_interval_s <= 0.0:
            return
        elapsed_s = now_wall_time_s - self._last_status_log_wall_time_s
        if elapsed_s < self._status_log_interval_s:
            return

        self._last_status_log_wall_time_s = now_wall_time_s
        stats = self.stats()
        self._node.get_logger().info(
            "Tracker health: "
            f"received={stats['received']}, "
            f"processed={stats['processed']}, "
            f"queue_drops={stats['queue_full_drops']}, "
            f"callback_failures={stats['callback_failures']}, "
            f"intentional_stride_skips={stats['stride_skips']}, "
            f"rate_limit_skips={stats['min_separation_skips']}, "
            f"queue_depth={stats['queue_depth']}/{stats['queue_size']}"
        )
