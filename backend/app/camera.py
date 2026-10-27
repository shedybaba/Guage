"""Camera motion: how far the background moved in every frame, and a way to take that out of tracked positions.

A handheld or bumped camera moves every pixel of the picture, so an object that is really still seems to move, and
a swinging object picks up the hand's motion on top of its own. Here the BACKGROUND is followed (corner-like points
away from the tracked objects, followed with optical flow), a shift-rotation-zoom is fitted to them for every frame
with outlier rejection (points on moving things are thrown out), and the tracked positions are moved back to where
they would be if the camera had stayed where it was in the first frame. Pure computer vision and arithmetic:
no model is involved.

Every frame is compared directly with a REFERENCE frame (the first one), not with the frame before it, because
chaining small steps adds their errors up (measured: five times worse over 150 frames). The reference is renewed only
when much of its background is no longer visible (a hand moves in, the view drifts), starting from where the old
reference placed that frame.

Only SHAKE-sized motion is corrected. A camera that pans far, zooms or turns a lot also changes how near and far
things shift against each other (parallax), and one background motion can then no longer be moved out of an object
in front of it: on a real handheld clip of a Newton's cradle (zoom up to 1.4x, turn up to 11 degrees) the correction
made the still balls move MORE. Such clips are reported but not corrected.

A steady camera is left alone: the correction is applied only when the background really moved (at least
`SIGNIFICANT_PX` somewhere in the picture) and the estimate is trustworthy (enough background points in nearly
every frame). Otherwise the estimate's own noise would be added to a good measurement.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

MAX_WIDTH = 320  # frames are shrunk to this width for the search (the result is converted back to full size)
MAX_POINTS = 400
MIN_POINTS = 12  # fewer surviving background points than this and a frame's estimate is not trusted
RANSAC_PX = 1.0  # in shrunk pixels: a point further than this from the fitted motion is not background
SIGNIFICANT_PX = 1.0  # the background must move at least this far (full-size pixels) before anything is corrected
MAX_ZOOM = 0.03  # beyond this zoom (3%) or turn the motion is not a shake and is not corrected
MAX_TURN_DEG = 4.0
USABLE_FRACTION = 0.95  # share of frames that need a trustworthy estimate
RENEW_BELOW = 0.6  # take a new reference when fewer than this share of the usual background points are left
MAX_BAD_RUN = 15  # this many untrusted frames in a row: the chain is lost and the whole estimate is not used


@dataclass
class CameraMotion:
    """`transforms[k]` maps a point of the FIRST frame to where the same background point is in frame k (full-size
    pixels, 3x3). Rows are NaN where nothing could be estimated and `usable` is False."""
    transforms: np.ndarray
    valid_fraction: float
    peak_shift_px: float  # the furthest any part of the picture was from its first-frame place
    usable: bool  # the estimate is trustworthy for (nearly) every frame
    significant: bool  # and the camera really moved, so the correction is worth applying
    frames: int = 0
    max_zoom: float = 0.0  # the biggest change of picture size against the first frame (0.03 = 3%)
    max_turn_deg: float = 0.0  # the biggest turn against the first frame

    @property
    def too_large(self):
        """A pan, zoom or turn too big to be a shake: not corrected (see the module text)."""
        return self.max_zoom > MAX_ZOOM or self.max_turn_deg > MAX_TURN_DEG

    @property
    def apply(self):
        return self.usable and self.significant and not self.too_large


def _free_mask(shape, boxes, scale):
    """255 where background points may be taken, 0 over the tracked objects (their boxes with some margin)."""
    mask = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        pad = 0.15 * max(x2 - x1, y2 - y1) + 4
        a, b = max(0, int((x1 - pad) * scale)), max(0, int((y1 - pad) * scale))
        c, d = int(np.ceil((x2 + pad) * scale)), int(np.ceil((y2 + pad) * scale))
        mask[b:d + 1, a:c + 1] = 0
    return mask


def _apply(m, pts):
    """Move Nx2 points with a 3x3 matrix."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    out = np.c_[pts, np.ones(len(pts))] @ m.T
    return out[:, :2]


def _fill_gaps(w):
    """Fill frames with no estimate from the frames around them (straight lines between the known matrices; the
    nearest known matrix at the ends). Only used when there are few such frames."""
    ok = np.isfinite(w[:, 0, 0])
    if ok.all() or not ok.any():
        return w
    out = w.copy()
    x = np.arange(len(w))
    for i in range(2):
        for j in range(3):
            out[:, i, j] = np.interp(x, x[ok], w[ok, i, j])
    out[:, 2, :] = [0.0, 0.0, 1.0]
    return out


SPIKE_PX = 2.0  # a frame whose background position is this far from the average of its two neighbours ...
SPIKE_TIMES = 8.0  # ... and this many times further than is usual for the clip is a one-frame glitch


def _drop_spikes(full, w0, h0):
    """Mark isolated one-frame glitches of the estimate (for example when something big enters the picture) as unknown.
    A real camera does not jump several pixels for one frame and come back."""
    ok = np.isfinite(full[:, 0, 0])
    if ok.sum() < 5:
        return full
    grid = np.array([[0, 0], [w0, 0], [0, h0], [w0, h0], [w0 / 2, h0 / 2]], float)
    pos = np.array([_apply(m, grid) if np.isfinite(m).all() else np.full((5, 2), np.nan) for m in full])
    dev = np.full(len(full), np.nan)
    dev[1:-1] = np.nanmax(np.hypot(*(pos[1:-1] - 0.5 * (pos[:-2] + pos[2:])).transpose(2, 0, 1)), axis=1)
    usual = np.nanmedian(dev)
    bad = dev > max(SPIKE_PX, SPIKE_TIMES * (usual if np.isfinite(usual) else 0.0))
    out = full.copy()
    out[bad] = np.nan
    return out


def estimate_camera_motion(path, boxes=None, expected_size=None, max_width=MAX_WIDTH, trace=None):
    """Follow the background through the video. `boxes` = {frame: [(x1, y1, x2, y2), ...]} of tracked objects (full-size
    pixels); `expected_size` = (width, height) the tracks are measured in. Returns CameraMotion, or None if the
    video cannot be read in that size. `trace`, a list, collects (frame, background points used) for debugging."""
    boxes = boxes or {}
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok:
            return None
        h0, w0 = frame.shape[:2]
        if expected_size and (w0, h0) != tuple(expected_size):
            return None
        s = min(1.0, max_width / w0)
        size = (max(8, round(w0 * s)), max(8, round(h0 * s)))
        s = size[0] / w0  # the exact scale actually used

        def prepare(f):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA) if size != (w0, h0) else g
            return cv2.GaussianBlur(g, (3, 3), 0)

        def detect(gray, k):
            mask = _free_mask(gray.shape, boxes.get(k, ()), s)
            return cv2.goodFeaturesToTrack(gray, MAX_POINTS, 0.01, 8, mask=mask)

        gray = prepare(frame)
        ref_gray, ref_pts = gray, detect(gray, 0)
        world_to_ref = np.eye(3)  # first frame -> the reference frame (shrunk pixels)
        b_prev = np.eye(3)  # reference -> previous frame: the starting guess for the next one
        healthy = 0  # background points that were still good soon after this reference was chosen
        w = [np.eye(3)]  # first frame -> frame k (shrunk pixels)
        bad_run = 0
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            k += 1
            gray = prepare(frame)
            estimate, inliers = None, 0
            if ref_pts is not None and len(ref_pts) >= MIN_POINTS:
                guess = _apply(b_prev, ref_pts).astype(np.float32).reshape(-1, 1, 2)
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref_gray, gray, ref_pts, guess, winSize=(21, 21), maxLevel=3,
                                                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                keep = st.ravel() == 1
                if keep.sum() >= MIN_POINTS:
                    src, dst = ref_pts[keep].reshape(-1, 2), nxt[keep].reshape(-1, 2)
                    free = _free_mask(gray.shape, boxes.get(k, ()), s)  # points now under a tracked object are not background
                    xi = np.clip(dst[:, 0].round().astype(int), 0, size[0] - 1)
                    yi = np.clip(dst[:, 1].round().astype(int), 0, size[1] - 1)
                    on_bg = free[yi, xi] > 0
                    src, dst = src[on_bg], dst[on_bg]
                    if len(src) >= MIN_POINTS:
                        m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
                                                             maxIters=2000, confidence=0.995, refineIters=10)
                        if m is not None and inl is not None and int(inl.sum()) >= MIN_POINTS:
                            estimate, inliers = np.vstack([m, [0, 0, 1]]), int(inl.sum())
            if estimate is None:
                bad_run += 1
                w.append(np.full((3, 3), np.nan))
                if bad_run > MAX_BAD_RUN:
                    break
                continue
            bad_run = 0
            b_prev = estimate
            w.append(estimate @ world_to_ref)
            healthy = max(healthy, inliers) if k < 5 or healthy == 0 else healthy
            if trace is not None:
                trace.append((k, inliers))
            if inliers < max(2 * MIN_POINTS, RENEW_BELOW * healthy):
                # most of the reference's points are gone (the scene changed or the view moved on): take a new
                # reference HERE, starting from where the old one put this frame
                fresh = detect(gray, k)
                if fresh is not None and len(fresh) >= 2 * MIN_POINTS:
                    ref_gray, ref_pts, world_to_ref, b_prev, healthy = gray, fresh, w[-1], np.eye(3), 0
    finally:
        cap.release()

    n_read = len(w)
    w = np.array(w)
    unit = np.diag([s, s, 1.0])
    full = np.linalg.inv(unit) @ w @ unit  # shrunk pixels -> full-size pixels
    full = _drop_spikes(full, w0, h0)
    valid = np.isfinite(full[:, 0, 0])
    valid_fraction = float(valid.mean())
    lost = bad_run > MAX_BAD_RUN
    usable = (not lost) and valid_fraction >= USABLE_FRACTION and n_read >= 2
    full = _fill_gaps(full)
    peak = 0.0
    if valid.any():
        grid = np.array([[x, y] for x in (0, w0 / 2, w0) for y in (0, h0 / 2, h0)], float)
        for m in full[valid]:
            peak = max(peak, float(np.max(np.hypot(*(_apply(m, grid) - grid).T))))
    zoom = turn = 0.0
    if valid.any():
        lin = full[valid][:, :2, :2]
        zoom = float(np.max(np.abs(np.sqrt(np.abs(np.linalg.det(lin))) - 1.0)))
        turn = float(np.max(np.abs(np.degrees(np.arctan2(lin[:, 1, 0], lin[:, 0, 0])))))
    return CameraMotion(full, valid_fraction, peak, usable, peak >= SIGNIFICANT_PX, n_read, zoom, turn)


def stabilize_tracks(tracks, motion):
    """Copies of the tracked points moved back to where they would be if the camera had stayed put (first frame).
    The original tracks (what is drawn over the video) are not touched."""
    inverse = np.linalg.inv(motion.transforms)
    out = {}
    for oid, pts in tracks.items():
        moved = []
        for p in pts:
            if p.frame >= len(inverse):
                moved.append(p)
                continue
            (nx, ny), = _apply(inverse[p.frame], [(p.cx, p.cy)])
            dx, dy = nx - p.cx, ny - p.cy
            moved.append(replace(p, cx=nx, cy=ny, x1=p.x1 + dx, x2=p.x2 + dx, y1=p.y1 + dy, y2=p.y2 + dy))
        out[oid] = moved
    return out
