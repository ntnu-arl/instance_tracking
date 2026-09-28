"""Tests for the workspace OpenLex pipeline helper utilities."""

import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pytest

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[4] / "scripts" / "openlex3d"),
)

import openlex_pipeline_utils as utils  # noqa: E402


class _FakeMetadata:
    def __init__(self, payload):
        self._payload = payload

    def get(self):
        return self._payload


@dataclass
class _FakeAttrs:
    open_vocab_features: np.ndarray
    mesh_connections: list[int]
    color: tuple[int, int, int]
    is_active: bool = True
    object_uid: int = 0
    source_track_ids: tuple[int, ...] = ()
    open_vocab_ignore_effective: bool = False
    metadata_payload: dict | None = None

    @property
    def metadata(self):
        return _FakeMetadata(self.metadata_payload or {})


@dataclass
class _FakeNode:
    id: str
    attributes: _FakeAttrs


@dataclass
class _FakeLayer:
    nodes: list[_FakeNode]


class _FakeMesh:
    def __init__(self, positions, labels=None, faces=None):
        self._positions = [np.asarray(position, dtype=np.float32) for position in positions]
        self._labels = list(labels or [utils.NO_TRACK] * len(self._positions))
        self._faces = list(faces or [])

    def num_vertices(self):
        return len(self._positions)

    def pos(self, index):
        return self._positions[index]

    def label(self, index):
        return self._labels[index]

    def num_faces(self):
        return len(self._faces)

    def face(self, index):
        return self._faces[index]


class _FakeGraph:
    def __init__(self, mesh, nodes):
        self.mesh = mesh
        self._layer = _FakeLayer(nodes)

    def has_mesh(self):
        return True

    def get_layer(self, _layer_id):
        return self._layer


class _FakeDsgModule:
    class DsgLayers:
        OBJECTS = "objects"


def _tracked_metadata(*, encoder_id="openclip:ViT-H-14:laion2b_s32b_b79k", object_uid=1, source_track_ids=(11,)):
    return {
        "hydra_tracked_object": {
            "enabled": True,
            "object_uid": object_uid,
            "source_track_ids": list(source_track_ids),
            "open_vocab_encoder_id": encoder_id,
        }
    }


def test_collect_scene_prediction_mean_normalizes_and_maps_points(monkeypatch):
    monkeypatch.setattr(utils, "load_spark_dsg", lambda: _FakeDsgModule())

    graph = _FakeGraph(
        _FakeMesh(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ),
        [
            _FakeNode(
                "O1",
                _FakeAttrs(
                    open_vocab_features=np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32),
                    mesh_connections=[0, 1],
                    color=(255, 0, 0),
                    metadata_payload=_tracked_metadata(object_uid=101, source_track_ids=(11, 22)),
                ),
            ),
            _FakeNode(
                "O2",
                _FakeAttrs(
                    open_vocab_features=np.zeros((2, 0), dtype=np.float32),
                    mesh_connections=[2],
                    color=(0, 255, 0),
                    metadata_payload=_tracked_metadata(object_uid=202),
                ),
            ),
            _FakeNode(
                "O3",
                _FakeAttrs(
                    open_vocab_features=np.array([[1.0], [1.0]], dtype=np.float32),
                    mesh_connections=[],
                    color=(0, 0, 255),
                    metadata_payload=_tracked_metadata(object_uid=303),
                ),
            ),
        ],
    )

    prediction = utils.collect_scene_prediction(
        graph,
        expected_encoder_id="openclip:ViT-H-14:laion2b_s32b_b79k",
    )

    assert prediction.points_xyz.shape == (2, 3)
    assert np.array_equal(prediction.point_object_indices, np.array([0, 0], dtype=np.int32))
    assert prediction.embeddings.shape == (1, 2)
    expected = np.array([1.5, 2.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(prediction.embeddings[0], expected)
    assert prediction.object_records[0]["object_uid"] == 101
    assert prediction.object_records[0]["source_track_ids"] == [11, 22]


def test_collect_scene_prediction_rejects_encoder_mismatch(monkeypatch):
    monkeypatch.setattr(utils, "load_spark_dsg", lambda: _FakeDsgModule())
    graph = _FakeGraph(
        _FakeMesh(positions=((0.0, 0.0, 0.0),)),
        [
            _FakeNode(
                "O1",
                _FakeAttrs(
                    open_vocab_features=np.array([[1.0], [0.0]], dtype=np.float32),
                    mesh_connections=[0],
                    color=(255, 0, 0),
                    metadata_payload=_tracked_metadata(
                        encoder_id="openclip:ViT-L-14:laion2b_s32b_b82k"
                    ),
                ),
            ),
        ],
    )

    with pytest.raises(ValueError, match="encoder_id"):
        utils.collect_scene_prediction(
            graph,
            expected_encoder_id="openclip:ViT-H-14:laion2b_s32b_b79k",
        )


def test_collect_scene_prediction_uses_dsg_mesh_connections_only(monkeypatch):
    monkeypatch.setattr(utils, "load_spark_dsg", lambda: _FakeDsgModule())
    graph = _FakeGraph(
        _FakeMesh(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            labels=(11, 11, utils.NO_TRACK),
        ),
        [
            _FakeNode(
                "O1",
                _FakeAttrs(
                    open_vocab_features=np.array([[1.0], [0.0]], dtype=np.float32),
                    mesh_connections=[0],
                    color=(255, 0, 0),
                    metadata_payload=_tracked_metadata(object_uid=101, source_track_ids=(11,)),
                ),
            ),
        ],
    )

    prediction = utils.collect_scene_prediction(
        graph,
        expected_encoder_id="openclip:ViT-H-14:laion2b_s32b_b79k",
    )

    assert prediction.points_xyz.shape == (1, 3)
    assert np.array_equal(prediction.point_object_indices, np.array([0], dtype=np.int32))
    assert np.allclose(
        prediction.points_xyz,
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert prediction.object_records[0]["mesh_connections"] == [0]


def test_build_overlay_payload_respects_stride_and_ignore_blend(monkeypatch):
    monkeypatch.setattr(utils, "load_spark_dsg", lambda: _FakeDsgModule())
    graph = _FakeGraph(
        _FakeMesh(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            labels=(11, 11, 22),
            faces=((0, 1, 2),),
        ),
        [
            _FakeNode(
                "O1",
                _FakeAttrs(
                    open_vocab_features=np.array([[1.0], [0.0]], dtype=np.float32),
                    mesh_connections=[0, 1],
                    color=(255, 0, 0),
                    metadata_payload=_tracked_metadata(object_uid=1, source_track_ids=(11,)),
                ),
            ),
            _FakeNode(
                "O2",
                _FakeAttrs(
                    open_vocab_features=np.array([[0.0], [1.0]], dtype=np.float32),
                    mesh_connections=[2],
                    color=(0, 255, 0),
                    metadata_payload={
                        "hydra_tracked_object": {
                            "enabled": True,
                            "object_uid": 2,
                            "source_track_ids": [22],
                            "open_vocab_ignore_effective": True,
                        }
                    },
                ),
            ),
        ],
    )

    payload = utils.build_overlay_payload(
        graph,
        config=utils.OverlayConfig(
            vertex_stride=2,
            unassigned_color=(90, 90, 90),
            unassigned_alpha=0.25,
        ),
    )

    assert payload["positions"].shape == (3, 3)
    assert payload["faces"].shape == (1, 3)
    assert tuple(payload["colors_rgba"][0][:3]) == (255, 0, 0)
    assert tuple(payload["colors_rgba"][1][:3]) == (90, 90, 90)
    assert tuple(payload["colors_rgba"][2][:3]) == tuple(
        utils.blend_color((0, 255, 0), (90, 90, 90), 0.85)
    )
