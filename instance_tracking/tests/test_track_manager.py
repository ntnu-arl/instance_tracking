import torch

from instance_tracking.tracking.track_manager import TrackManager, TrackManagerConfig


def _make_manager(**overrides) -> TrackManager:
    config = TrackManagerConfig(**overrides)
    return TrackManager(config=config, device=torch.device("cpu"))


def test_iou_gate_blocks_zero_overlap_feature_match() -> None:
    manager = _make_manager(
        min_iou_gate=0.15,
        eligibility_factor=0.0,
        alignment_threshold=0.35,
    )
    features = torch.ones((4, 4, 4), dtype=torch.float32)

    initial_masks = torch.zeros((1, 4, 4), dtype=torch.bool)
    initial_masks[0, :2, :2] = True
    manager.initialize_from_masks(initial_masks, features, (4, 4))

    new_masks = torch.zeros((1, 4, 4), dtype=torch.bool)
    new_masks[0, 2:, 2:] = True

    matched, new, retired, _ = manager.update_from_keyframe(new_masks, features, (4, 4))

    assert matched == []
    assert len(new) == 1
    assert len(retired) == 1
    assert retired[0].is_terminated
    assert manager.active_tracks == new


def test_ineligible_propagated_track_is_retired_at_keyframe() -> None:
    manager = _make_manager(
        min_iou_gate=0.0,
        eligibility_factor=0.5,
        alignment_threshold=0.35,
    )
    features = torch.ones((4, 4, 4), dtype=torch.float32)

    initial_masks = torch.zeros((1, 4, 4), dtype=torch.bool)
    initial_masks[0, :2, :2] = True
    created_tracks = manager.initialize_from_masks(initial_masks, features, (4, 4))
    track = created_tracks[0]
    assert track.last_keyframe_mask_area == 4

    tiny_propagated_mask = torch.zeros((4, 4), dtype=torch.float32)
    tiny_propagated_mask[0, 0] = 1.0
    track.update_propagated(tiny_propagated_mask)
    assert track.current_mask_area == 1

    new_masks = torch.zeros((1, 4, 4), dtype=torch.bool)
    new_masks[0, 0, 0] = True

    matched, new, retired, _ = manager.update_from_keyframe(new_masks, features, (4, 4))

    assert matched == []
    assert len(new) == 1
    assert len(retired) == 1
    assert retired[0].track_id == track.track_id
    assert retired[0].is_terminated
    assert manager.active_tracks == new


def test_matched_keyframe_updates_prototype_with_ema_and_keeps_color() -> None:
    manager = _make_manager(
        min_iou_gate=0.0,
        eligibility_factor=0.0,
        alignment_threshold=0.35,
        prototype_ema_alpha=0.25,
    )
    initial_features = torch.zeros((2, 2, 4), dtype=torch.float32)
    initial_features[..., 0] = 1.0

    initial_masks = torch.ones((1, 2, 2), dtype=torch.bool)
    created_tracks = manager.initialize_from_masks(initial_masks, initial_features, (2, 2))
    track = created_tracks[0]
    initial_prototype = track.prototype.clone()
    initial_color = track.color

    propagated_mask = torch.ones((2, 2), dtype=torch.float32)
    track.update_propagated(propagated_mask)
    assert torch.allclose(track.prototype, initial_prototype)
    assert track.color == initial_color

    keyframe_features = torch.zeros((2, 2, 4), dtype=torch.float32)
    keyframe_features[..., 1] = 1.0
    new_masks = torch.ones((1, 2, 2), dtype=torch.bool)

    matched, new, retired, _ = manager.update_from_keyframe(new_masks, keyframe_features, (2, 2))

    assert len(matched) == 1
    assert new == []
    assert retired == []
    assert matched[0] is track
    expected = torch.tensor([0.75, 0.25, 0.0, 0.0], dtype=torch.float32)
    expected = expected / expected.norm(p=2)
    assert torch.allclose(track.prototype, expected)
    assert track.color == initial_color
