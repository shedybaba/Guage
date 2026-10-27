"""Camera motion: a rendered clip with a KNOWN shaking camera and a KNOWN pendulum; the correction must find the truth."""
import cv2
import numpy as np
import pytest

from app.camera import estimate_camera_motion, stabilize_tracks
from app.sam import TrackPoint

W, H, FPS, N = 480, 360, 30, 150


def textured_world(seed=3, size=(1100, 900)):
    """A big picture with plenty of corners to follow (blurred random blobs and lines)."""
    rng = np.random.default_rng(seed)
    img = np.full((size[1], size[0]), 120, np.uint8)
    for _ in range(500):
        x, y, r = int(rng.integers(0, size[0])), int(rng.integers(0, size[1])), int(rng.integers(3, 18))
        cv2.circle(img, (x, y), r, int(rng.integers(20, 235)), -1)
    for _ in range(80):
        a, b = (int(rng.integers(0, size[0])), int(rng.integers(0, size[1]))), (int(rng.integers(0, size[0])), int(rng.integers(0, size[1])))
        cv2.line(img, a, b, int(rng.integers(20, 235)), 2)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def bob_world_position(i):
    """The pendulum's TRUE position in world pixels: period 1.2 s, amplitude 60 px."""
    t = i / FPS
    return 550 + 60 * np.cos(2 * np.pi * t / 1.2), 420 + 0.0 * t


def camera_offset(i, kind):
    """How far the camera is from its first-frame place (translation + a small rotation), in pixels and degrees."""
    t = i / FPS
    if kind == "still":
        return 0.0, 0.0, 0.0
    return 7 * np.sin(2 * np.pi * 0.7 * t) + 0.04 * i, 5 * np.sin(2 * np.pi * 1.3 * t + 1) - 0.02 * i, 0.6 * np.sin(2 * np.pi * 0.5 * t)


def render(path, kind, hand=False):
    """Write the clip and return the true camera matrices (world -> frame) and the bob's frame positions."""
    base_world = textured_world()
    handtex = np.random.default_rng(5).integers(0, 255, (180, 300), dtype=np.uint8)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    truth, centres = [], []
    base = np.array([[1, 0, -300.0], [0, 1, -250.0]])  # the camera looks at a window of the world
    for i in range(N):
        world = base_world
        if hand and 40 <= i < 110:  # something big and textured crosses the scene, as a hand does
            world = base_world.copy()
            x = int(300 + (i - 40) * 6)
            world[380:560, x:x + 300] = handtex
        dx, dy, deg = camera_offset(i, kind)
        rot = cv2.getRotationMatrix2D((550, 420), deg, 1.0)
        m = np.vstack([rot, [0, 0, 1]])
        shift = np.array([[1, 0, base[0, 2] + dx], [0, 1, base[1, 2] + dy], [0, 0, 1]])
        cam = shift @ m  # world -> frame
        frame = cv2.warpAffine(world, cam[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bx, by = bob_world_position(i)
        cx, cy = (cam @ np.array([bx, by, 1.0]))[:2]
        cv2.circle(frame, (int(round(cx)), int(round(cy))), 14, 255, -1)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        truth.append(cam)
        centres.append((cx, cy))
    out.release()
    return np.array(truth), np.array(centres)


def tracks_from(centres):
    return {"0": [TrackPoint(i, "0", x - 14, y - 14, x + 14, y + 14, 0.9, float(x), float(y), 600.0) for i, (x, y) in enumerate(centres)]}


def boxes_from(tracks):
    return {p.frame: [(p.x1, p.y1, p.x2, p.y2)] for p in tracks["0"]}


@pytest.fixture(scope="module")
def shaky(tmp_path_factory):
    path = tmp_path_factory.mktemp("cam") / "shaky.mp4"
    truth, centres = render(path, "shaky")
    return path, truth, centres


def test_shaking_camera_is_followed_and_the_bob_goes_back_to_its_true_path(shaky):
    path, truth, centres = shaky
    tracks = tracks_from(centres)
    motion = estimate_camera_motion(path, boxes_from(tracks), expected_size=(W, H))
    assert motion.usable and motion.significant and motion.valid_fraction == 1.0
    assert motion.peak_shift_px > 8  # the shake really was that big
    fixed = stabilize_tracks(tracks, motion)["0"]
    # the corrected path is the pendulum's own path seen from the first frame's camera position
    first = truth[0]
    want = np.array([(first @ np.array([*bob_world_position(i), 1.0]))[:2] for i in range(N)])
    got = np.array([(p.cx, p.cy) for p in fixed])
    err = np.hypot(*(got - want).T)
    assert err.mean() < 0.1 and err.max() < 0.3  # measured: 0.02 and 0.07 px
    # while the uncorrected path was off by many pixels
    raw = np.hypot(*(centres - want).T)
    assert raw.max() > 8


def test_the_original_tracks_are_not_changed(shaky):
    path, _, centres = shaky
    tracks = tracks_from(centres)
    before = [(p.cx, p.cy) for p in tracks["0"]]
    motion = estimate_camera_motion(path, boxes_from(tracks), expected_size=(W, H))
    stabilize_tracks(tracks, motion)
    assert [(p.cx, p.cy) for p in tracks["0"]] == before


def test_a_steady_camera_is_left_alone(tmp_path):
    path = tmp_path / "still.mp4"
    _, centres = render(path, "still")
    tracks = tracks_from(centres)
    motion = estimate_camera_motion(path, boxes_from(tracks), expected_size=(W, H))
    assert motion.usable and not motion.significant and not motion.apply
    assert motion.peak_shift_px < 0.5


def test_a_picture_with_nothing_to_follow_is_not_used(tmp_path):
    path = tmp_path / "blank.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(40):
        out.write(np.full((H, W, 3), 128, np.uint8))
    out.release()
    motion = estimate_camera_motion(path, {}, expected_size=(W, H))
    assert not motion.usable and not motion.apply


def test_a_video_of_another_size_is_refused(shaky):
    path, _, _ = shaky
    assert estimate_camera_motion(path, {}, expected_size=(W + 1, H)) is None


def test_a_big_pan_or_zoom_is_reported_but_not_corrected():
    import numpy as np
    from app.camera import CameraMotion
    n = 50
    w = np.tile(np.eye(3), (n, 1, 1))
    w[:, 0, 0] = w[:, 1, 1] = np.linspace(1.0, 1.2, n)  # a 20% zoom
    w[:, 0, 2] = np.linspace(0, 300, n)
    m = CameraMotion(w, 1.0, 300.0, True, True, n, 0.2, 0.0)
    assert m.too_large and not m.apply
    shake = CameraMotion(np.tile(np.eye(3), (n, 1, 1)), 1.0, 5.0, True, True, n, 0.002, 0.5)
    assert shake.apply


def test_the_shake_regime_is_measured_from_the_matrices(shaky):
    path, _, centres = shaky
    m = estimate_camera_motion(path, boxes_from(tracks_from(centres)), expected_size=(W, H))
    assert 0.0 < m.max_turn_deg < 1.0 and m.max_zoom < 0.01 and not m.too_large  # the render turns by up to 0.6 degrees


def test_a_hand_crossing_the_scene_does_not_throw_the_estimate_off(tmp_path):
    path = tmp_path / "hand.mp4"
    truth, centres = render(path, "shaky", hand=True)
    tracks = tracks_from(centres)
    motion = estimate_camera_motion(path, boxes_from(tracks), expected_size=(W, H))
    assert motion.usable and motion.apply
    fixed = stabilize_tracks(tracks, motion)["0"]
    want = np.array([(truth[0] @ np.array([*bob_world_position(i), 1.0]))[:2] for i in range(N)])
    err = np.hypot(*(np.array([(p.cx, p.cy) for p in fixed]) - want).T)
    assert err.mean() < 0.15 and err.max() < 0.5
