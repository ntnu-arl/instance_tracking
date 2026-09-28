import torch

from instance_tracking.models.mask_refinement import MaskRefinementConfig, MaskRefiner


def test_refiner_returns_confidence_of_selected_output_label() -> None:
    refiner = MaskRefiner(
        MaskRefinementConfig(
            enabled=True,
            stronger_on_keyframes=False,
            smooth_probs=False,
            post_smooth=False,
            morph_open=False,
            morph_close=False,
            threshold=0.5,
            resolve_overlaps=True,
            overlap_strategy="confidence",
        ),
        torch.device("cpu"),
    )

    probs = torch.tensor(
        [
            [[0.6, 0.4, 0.0], [0.1, 0.7, 0.2]],
            [[0.1, 0.2, 0.8], [0.0, 0.6, 0.9]],
        ],
        dtype=torch.float32,
    )
    result = refiner.refine_with_confidence(
        probs=probs,
        output_shape=(2, 2),
        instance_ids=torch.tensor([7, 9], dtype=torch.int32),
    )

    expected_mask = torch.tensor([[0, 7], [9, 9]], dtype=torch.uint16)
    expected_confidence = torch.tensor(
        [[0.0, 0.7], [0.8, 0.9]],
        dtype=torch.float32,
    )

    assert torch.equal(result.mask_image.cpu(), expected_mask)
    assert torch.equal(result.label_confidence.cpu(), expected_confidence)
