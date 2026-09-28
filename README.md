# Instance Tracking 

This repository provides the image-space tracker used by [TʀᴀᴄᴋGʀᴀᴘʜ](https://github.com/ntnu-arl/trackgraph).
FastSAM masks and CLIP features are computed at sparse keyframes, while masks and
their identities are propagated between keyframes using dense DINOv3 features.

The source tracks provide consistent masks and aggregated CLIP features for
online 3D mapping.

## Install

Follow the [TʀᴀᴄᴋGʀᴀᴘʜ setup guide](https://github.com/ntnu-arl/trackgraph_ros/tree/main#setup)
to import the required repositories, build the ROS packages, and
create a Python virtual environment. From that workspace, with the environment
active, install this package's Python dependencies:

```bash
python -m pip install -e "src/instance_tracking/instance_tracking[open_vocab]"
```

The tracker requires this DINOv3 ViT-S+/16 checkpoint as a local file:

```text
<workspace>/models/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

Follow the official
[DINOv3 pretrained-model instructions](https://github.com/facebookresearch/dinov3#pretrained-models)
and request the weights through the linked
[Meta DINOv3 access form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
Download the LVD-1689M ViT-S+/16 checkpoint with `wget`, as recommended upstream,
and save it under the exact filename above. Set `DINOV3_WEIGHTS_PATH` only when the
DINOv3 weights directory lives elsewhere. The loader fetches and caches the DINOv3
architecture code through `torch.hub` on first use, but it does not download this
checkpoint.

`FastSAM-x.pt` does not normally need to be installed manually. Ultralytics downloads
the recognized checkpoint on first use when it is absent. OpenCLIP likewise downloads
the selected pretrained checkpoint into the Torch/Hugging Face cache on first use.
Both automatic downloads require network access and a writable cache or working
directory; for offline deployment, pre-populate those files and caches.

## Launch

The generic launch uses the `dev` preset:

```bash
ros2 launch instance_tracking_ros instance_tracking.launch.yaml
```

Select another installed preset by name:

```bash
ros2 launch instance_tracking_ros instance_tracking.launch.yaml \
  config_name:=deployment
```

An explicit file remains supported:

```bash
ros2 launch instance_tracking_ros instance_tracking.launch.yaml \
  config_path:=/absolute/path/to/tracker.yaml
```

For the uHumans2 office test, use the dataset wrapper:

```bash
ros2 launch instance_tracking_ros uhumans2.launch.yaml
```

It selects `dev.yaml`, uses namespace `tesse/left_cam`, and consumes the original bag
topics without remapping. The same uHumans2 topic contract can be tested with the
resource-conscious deployment preset:

```bash
ros2 launch instance_tracking_ros uhumans2.launch.yaml \
  config_name:=deployment
```

## Robot

Edit the camera topic defaults at the top of
[`robot.launch.yaml`](instance_tracking_ros/launch/robot.launch.yaml), or pass them
as arguments. They must match the mapper's `trackgraph_robot.launch.yaml`:

```bash
ros2 launch instance_tracking_ros robot.launch.yaml \
  rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/aligned_depth_to_color/image_raw
```

This selects `deployment`. The default `tracker_namespace:=/front_camera`
publishes under `/front_camera/tracking`, matching TʀᴀᴄᴋGʀᴀᴘʜ.

## License

Released under BSD-3-Clause.
