"""From a checked plan and tracked objects to a measurement: scale, tool dispatch, and the payloads the API returns.

Scale: free fall and velocity need metres per pixel. The user states the real length of something in the picture
(a metre stick, A4 paper); SAM finds that object, and its length along its own main axis in pixels gives the scale.
The uncertainty of that scale is carried into the answer (it is usually the biggest error), and a reference that is
small in the picture or changes length from frame to frame widens it.
"""
import csv
import io

import numpy as np

from . import signals
from .quality import LogEntry, track_with_checks
from .store import jsonable
from .tools.collision import collision_momentum
from .tools.counting import count_objects, line_crossings
from .tools.growth import growth_over_time
from .tools.kinematics import free_fall, velocity_acceleration
from .tools.pendulum import pendulum_period
from .tools.rotation import rotation_rate
from .tools.spring import oscillation_frequency

SCALE_NEEDED = ("free_fall", "velocity_acceleration")
SCALE_EDGE_FLOOR = 0.01  # the outline of a reference object is good to about 1%
SMALL_REFERENCE = 0.10  # a reference shorter than 10% of the picture makes a poor ruler


class NeedsReferenceObject(Exception):
    """A scale was asked for (a real length was given) but no object in the picture was named to measure it from."""


class MeasureError(Exception):
    """The measurement could not be made; the message is for the user."""


class ScaleInfo:
    def __init__(self, m_per_px, rel_uncertainty, notes, frames_billed=0, log=()):
        self.m_per_px, self.rel_uncertainty, self.notes = m_per_px, rel_uncertainty, notes
        self.frames_billed, self.log = frames_billed, list(log)


async def measure_scale(tracker, plan, info, known_values, on_log=None):
    """Metres per pixel, or None if this measurement does not need (and was not given) a scale."""
    known = known_values or {}
    if known.get("scale_m_per_px"):
        rel = known.get("scale_uncertainty_fraction")
        return ScaleInfo(float(known["scale_m_per_px"]), float(rel) if rel is not None else None,
                         ["Using the scale you gave (metres per pixel)."])
    ref = plan.values.get("reference_length")
    if ref is None:
        if plan.tool in SCALE_NEEDED:
            raise MeasureError("No known length was given, so metres cannot be worked out.")
        return None
    if not plan.reference_phrases:
        if plan.tool in SCALE_NEEDED:
            raise NeedsReferenceObject()
        return None  # a length was stated but no object named: this measurement can go without a scale
    checked = await track_with_checks(tracker, plan.reference_phrases, info.frame_count, on_log=on_log)
    if not checked.points:
        raise MeasureError(f"Gauge could not find the reference object ({plan.reference_phrases[0]}) in the picture, "
                           "so it cannot turn pixels into metres. Make sure it is fully visible.")
    sides = np.array([p.long_side or max(p.x2 - p.x1, p.y2 - p.y1) for p in checked.points], float)
    med = float(np.median(sides))
    spread = float(1.4826 * np.median(np.abs(sides - med)) / med) if med > 0 else 0.0
    rel = float(np.sqrt(SCALE_EDGE_FLOOR ** 2 + (1.0 / med) ** 2 + spread ** 2))
    notes = [f"Scale from the {plan.reference_phrases[0]}: {ref.said} is {med:.0f} pixels long in the picture "
             f"(the object's length along its own main direction), so one pixel is {ref.value / med * 1000:.2f} mm."]
    if med < SMALL_REFERENCE * max(info.width, info.height):
        notes.append("The reference object is small in the picture, so the scale is less accurate; film with it larger.")
    if spread > 0.03:
        notes.append("The reference object's length changed noticeably between frames (it moved or tilted); "
                     "the scale uncertainty is widened for that.")
    notes.append("The reference must be in the same plane as the motion (the same distance from the camera).")
    return ScaleInfo(ref.value / med, rel, notes, checked.frames_billed, checked.log)


# tools whose answer is spoiled when the camera moves; counting and growth do not depend on exact positions, and the
# collision tool already removes camera motion itself (using the objects that stay still), which is calibrated
CAMERA_TOOLS = {"pendulum_period", "oscillation_frequency", "free_fall", "velocity_acceleration", "rotation_rate",
                "line_crossings"}


def camera_log_entry(motion):
    """The line in 'how Gauge checked its work' about the camera (None if nothing could be said)."""
    if motion is None:
        return None
    if not motion.usable:
        return LogEntry("camera", "Camera check: there was not enough steady background to follow, so the camera's motion "
                                  "could not be measured. If the camera moved, positions and speeds are less reliable.", ok=None)
    if not motion.significant:
        return LogEntry("camera", f"Camera check: the background stayed still (the most it moved was {motion.peak_shift_px:.1f} px), "
                                  "so nothing was corrected.", ok=True)
    if motion.too_large:
        return LogEntry("camera", f"Camera check: the camera moved a lot (the background shifted by up to {motion.peak_shift_px:.0f} px, "
                                  "with zoom or turning). That is more than a shake, and nearer and farther things shift differently, "
                                  "so Gauge did not try to correct it. Positions, speeds and accelerations are less reliable: "
                                  "film again with the camera propped still.", ok=False)
    return LogEntry("camera", f"Camera check: the camera moved (the background shifted by up to {motion.peak_shift_px:.0f} px). "
                              "Gauge followed the background and moved every tracked position back to where it would be "
                              "if the camera had stayed still.", ok=True)


def values_si(plan):
    return {k: e.value for k, e in plan.values.items()}


def timelapse_speedup(values, timestamps):
    """Real seconds that pass per second of video, from what the user said about the time-lapse (or None).

    `values` are SI values (seconds). "The video covers 3 days" gives 3 days / the video's length; "a photo every
    10 minutes" gives 10 minutes / the time between frames.
    """
    t = np.asarray(timestamps, float)
    if len(t) < 2:
        return None
    if "real_duration" in values and t[-1] > t[0]:
        return values["real_duration"] / float(t[-1] - t[0])
    if "frame_interval" in values:
        return values["frame_interval"] / float(np.median(np.diff(t)))
    return None


def size_series(points, object_id, timestamps, quantity):
    """Time and size of one tracked object in every frame: its box height or width, or its mask area (pixels)."""
    pts = sorted((p for p in points if p.object_id == object_id and p.frame < len(timestamps)), key=lambda p: p.frame)
    t = np.array([timestamps[p.frame] for p in pts])
    pick = {"height": lambda p: p.y2 - p.y1, "width": lambda p: p.x2 - p.x1}.get(quantity, lambda p: p.area)
    return t, np.array([pick(p) for p in pts], float)


def run_tool(plan, checked, info, scale):
    """Run the planned tool on the tracked objects. `scale` is a ScaleInfo or None."""
    v = values_si(plan)
    points = [p for pts in checked.tracks.values() for p in pts]
    size = (info.width, info.height)
    m_per_px = scale.m_per_px if scale else None
    rel = scale.rel_uncertainty if scale else None
    if plan.tool in ("pendulum_period", "oscillation_frequency", "free_fall", "velocity_acceleration"):
        main = signals.pick_main_object(points)
        s = signals.series_for_object(points, main, info.timestamps)
        if plan.tool == "pendulum_period":
            return pendulum_period(s["t"], s["x"], s["y"], length_m=v.get("string_length"))
        if plan.tool == "oscillation_frequency":
            return oscillation_frequency(s["t"], s["x"], s["y"], mass_kg=v.get("mass"), spring_mass_kg=v.get("spring_mass"),
                                         scale_m_per_px=m_per_px)
        if plan.tool == "free_fall":
            return free_fall(s["t"], s["x"], s["y"], scale_m_per_px=m_per_px, scale_uncertainty_fraction=rel)
        return velocity_acceleration(s["t"], s["x"], s["y"], scale_m_per_px=m_per_px, scale_uncertainty_fraction=rel,
                                     quantity=plan.quantity, at_time=v.get("time"), ramp_angle_deg=v.get("ramp_angle"))
    if plan.tool in ("growth_over_time", "rotation_rate"):
        main = signals.pick_main_object(points)
        if plan.tool == "rotation_rate":
            s = signals.series_for_object(points, main, info.timestamps)
            return rotation_rate(s["t"], s["x"], s["y"])
        t, size = size_series(points, main, info.timestamps, plan.quantity)
        return growth_over_time(t, size, plan.quantity or "area", scale_m_per_px=m_per_px, scale_uncertainty_fraction=rel,
                                speedup=timelapse_speedup(v, info.timestamps),
                                coverage=len(size) / max(info.frame_count, 1))
    if plan.tool == "collision_momentum":
        t, pos, sizes = signals.positions_from_tracks(checked.tracks, info.timestamps)
        masses = {k: v["mass"] for k in pos} if "mass" in v else None
        return collision_momentum(t, pos, masses=masses, scale_m_per_px=m_per_px, sizes=sizes)
    if plan.tool == "count_objects":
        return count_objects(checked.tracks, info.timestamps, size)
    if plan.tool == "line_crossings":
        line = {"orientation": plan.line["orientation"], "position": plan.line["position_px"]}
        return line_crossings(checked.tracks, info.timestamps, size, line=line, check_camera=plan.check_camera)
    raise MeasureError(f"Gauge has no way to run '{plan.tool}'.")


# ---------------------------------------------------------------- what the API returns

def tracks_payload(tracks, info):
    """Final tracks (ids joined, lost frames re-acquired) in a compact form for the overlay player."""
    objects = {}
    for oid, pts in tracks.items():
        pts = sorted(pts, key=lambda p: p.frame)
        objects[str(oid)] = {
            "frames": [p.frame for p in pts],
            "t": [round(float(info.timestamps[p.frame] - info.timestamps[0]), 4) for p in pts],
            "box": [[round(p.x1, 1), round(p.y1, 1), round(p.x2, 1), round(p.y2, 1)] for p in pts],
            "centre": [[round(p.cx, 1), round(p.cy, 1)] for p in pts],
            "confidence": [None if p.confidence is None else round(p.confidence, 3) for p in pts]}
    return jsonable({"width": info.width, "height": info.height, "fps": info.fps, "frame_count": info.frame_count,
                     "objects": objects})


def tracks_csv(tracks, info):
    """The tracked data as CSV: one row per object per frame."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["frame", "time_s", "object_id", "x_px", "y_px", "box_x1", "box_y1", "box_x2", "box_y2", "confidence",
                "area_px"])
    rows = [(p.frame, oid, p) for oid, pts in tracks.items() for p in pts]
    for frame, oid, p in sorted(rows, key=lambda r: (r[0], str(r[1]))):
        w.writerow([frame, f"{info.timestamps[frame] - info.timestamps[0]:.4f}", oid, f"{p.cx:.2f}", f"{p.cy:.2f}",
                    f"{p.x1:.1f}", f"{p.y1:.1f}", f"{p.x2:.1f}", f"{p.y2:.1f}",
                    "" if p.confidence is None else f"{p.confidence:.3f}", f"{p.area:.0f}"])
    return out.getvalue()


def fact_dict(f):
    return {"id": f.id, "label": f.label, "display": f.display, "value": f.value, "uncertainty": f.uncertainty,
            "units": f.units}
