import carla
import math
import heapq
import queue
import threading
import time
import random

import cv2
import numpy as np
from flask import Flask, Response, jsonify

# import occupancy grid function
from occupancy_grid import (
    build_semantic_grid as build_semantic_grid_truth,
    collect_lot_semantics,
    STATIC_CAR,
    DYNAMIC_CAR,
    ROAD_LINE,
    LOT_CENTER_X as OCC_LOT_CENTER_X,
    LOT_CENTER_Y as OCC_LOT_CENTER_Y,
    LOT_WIDTH_M as OCC_LOT_WIDTH_M,
    LOT_HEIGHT_M as OCC_LOT_HEIGHT_M,
    RESOLUTION as OCC_RESOLUTION,
)

# =========================
# CONFIG
# =========================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5000

MAP_NAME = "Town05"

GRID_RES = 0.5
EGO_CLEARANCE_MARGIN_M = 0.20
INFLATION_RADIUS_M = None

FREE = 0
OCCUPIED = 1
INFLATED = 2
LINE_OCCUPIED = 3 # parking lines: blocked, but not inflated

# =========================
# SEMANTIC GRID LABELS
# =========================
SEM_OUTSIDE     = 0
SEM_DRIVABLE    = 1
SEM_STATIC_CAR  = 2
SEM_DYNAMIC_CAR = 3
SEM_ROAD_LINE   = 4

# =========================
# LOT DEFINITION
# must match occupancy_grid.py
# =========================
LOT_CENTER_X = -95.24 + 85   # -10.24
LOT_CENTER_Y = -88.04 + 60   # -28.04
LOT_WIDTH_M  = 60.0
LOT_HEIGHT_M = 60.0

GRID_W = int(LOT_WIDTH_M  / GRID_RES)   # 120
GRID_H = int(LOT_HEIGHT_M / GRID_RES)   # 120

ORIGIN_X = LOT_CENTER_X - LOT_WIDTH_M  / 2   # -40.24
ORIGIN_Y = LOT_CENTER_Y - LOT_HEIGHT_M / 2   # -58.04

GOAL_REACHED_DIST = 0.4

assert abs(LOT_CENTER_X - OCC_LOT_CENTER_X) < 1e-6
assert abs(LOT_CENTER_Y - OCC_LOT_CENTER_Y) < 1e-6
assert abs(LOT_WIDTH_M - OCC_LOT_WIDTH_M) < 1e-6
assert abs(LOT_HEIGHT_M - OCC_LOT_HEIGHT_M) < 1e-6
assert abs(GRID_RES - OCC_RESOLUTION) < 1e-6

# =========================
# STATIC CAMERA
# =========================
# CAMERA_X     = -45.0
# CAMERA_Y     = -25.0
CAMERA_X     = -10.24086
CAMERA_Y     = -28.037468
CAMERA_Z     =  55.0
CAM_IMG_W    = 1000
CAM_IMG_H    =  700
CAM_HFOV_DEG =  90.0

_half_hfov = math.radians(CAM_HFOV_DEG / 2)
CAM_SPAN_Y = 2 * CAMERA_Z * math.tan(_half_hfov)
CAM_SPAN_X = 2 * CAMERA_Z * math.tan(
    math.atan(math.tan(_half_hfov) * CAM_IMG_H / CAM_IMG_W))

# =========================
# SPAWN
# =========================
SPAWN_X   = -35.21
SPAWN_Y   = -28.03
SPAWN_YAW =  90.0

# =========================
# SHARED STATE
# =========================
demo_state = {
    "phase": "waiting",
    "message": "Ready",
    "slot_id": None,
    "is_parked": False,
}
selected_slot = {"slot": None}
last_parked_slot = {"slot": None}
image_queue = queue.Queue(maxsize=1)
run_lock = threading.Lock()

# globals updated at runtime
semantic_grid = None
nav_grid = None
inflated_grid = None
slots_data = []
road_line_bbs_cache = []


# =========================
# GEOMETRY HELPERS
# =========================
def pixel_to_world(px, py):
    wx = CAMERA_X + (0.5 - py / CAM_IMG_H) * CAM_SPAN_X
    wy = CAMERA_Y + (px / CAM_IMG_W - 0.5) * CAM_SPAN_Y
    return round(wx, 2), round(wy, 2)

def world_to_grid(x, y, ox, oy):
    return int((x - ox) / GRID_RES), int((y - oy) / GRID_RES)

def grid_to_world(gx, gy, ox, oy):
    return ox + (gx + 0.5) * GRID_RES, oy + (gy + 0.5) * GRID_RES

def in_bounds(gx, gy):
    return 0 <= gx < GRID_W and 0 <= gy < GRID_H

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def normalize_angle_deg(a):
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a

def get_speed(v):
    vel = v.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def bbox_center_in_lot(bb, margin=2.0):
    x = bb.location.x
    y = bb.location.y
    return (
        ORIGIN_X - margin <= x <= ORIGIN_X + LOT_WIDTH_M + margin
        and
        ORIGIN_Y - margin <= y <= ORIGIN_Y + LOT_HEIGHT_M + margin
    )


# =========================
# DRAW HELPERS
# =========================
def draw_path(world_obj, path, color, z=0.5):
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        world_obj.debug.draw_line(
            carla.Location(x=x1, y=y1, z=z),
            carla.Location(x=x2, y=y2, z=z),
            thickness=0.12,
            color=color,
            life_time=120.0
        )

def draw_label(world_obj, x, y, text, color):
    world_obj.debug.draw_string(
        carla.Location(x=x, y=y, z=1.5),
        text,
        draw_shadow=False,
        color=color,
        life_time=120.0
    )


# =========================
# GRID RASTERIZATION
# =========================
def _rasterize_bbox_semantic(grid, bb_location, bb_extent, val, shrink_x=0.0, shrink_y=0.0):
    ex = max(0.02, bb_extent.x - shrink_x)
    ey = max(0.02, bb_extent.y - shrink_y)

    min_x = bb_location.x - ex
    max_x = bb_location.x + ex
    min_y = bb_location.y - ey
    max_y = bb_location.y + ey

    gx0, gy0 = world_to_grid(min_x, min_y, ORIGIN_X, ORIGIN_Y)
    gx1, gy1 = world_to_grid(max_x, max_y, ORIGIN_X, ORIGIN_Y)

    gx_min, gx_max = sorted([gx0, gx1])
    gy_min, gy_max = sorted([gy0, gy1])

    gx_min = max(gx_min, 0)
    gx_max = min(gx_max, GRID_W - 1)
    gy_min = max(gy_min, 0)
    gy_max = min(gy_max, GRID_H - 1)

    if gx_min <= gx_max and gy_min <= gy_max:
        grid[gy_min:gy_max + 1, gx_min:gx_max + 1] = val


def draw_rotated_box(grid, cx, cy, yaw_rad, hl, hw, ox, oy, val):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    corners = [
        (cx + dx * c - dy * s, cy + dx * s + dy * c)
        for dx, dy in [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]

    mgx, mgy = world_to_grid(min(xs), min(ys), ox, oy)
    Mgx, Mgy = world_to_grid(max(xs), max(ys), ox, oy)

    for gx in range(max(0, mgx), min(GRID_W, Mgx + 1)):
        for gy in range(max(0, mgy), min(GRID_H, Mgy + 1)):
            wx, wy = grid_to_world(gx, gy, ox, oy)
            lx = (wx - cx) * c + (wy - cy) * s
            ly = -(wx - cx) * s + (wy - cy) * c
            if abs(lx) <= hl and abs(ly) <= hw:
                grid[gy, gx] = val


def build_nav_grid_from_semantics(sem):
    """
    Navigation grid:
    - lot interior starts FREE
    - cars are OCCUPIED (and later inflated)
    - parking lines are LINE_OCCUPIED (blocked, but not inflated)
    """
    nav = np.full((GRID_H, GRID_W), FREE, dtype=np.uint8)

    # hard obstacles that should be inflated
    nav[sem == STATIC_CAR] = OCCUPIED
    nav[sem == DYNAMIC_CAR] = OCCUPIED
    # painted parking lines: blocked, but no inflation
    nav[sem == ROAD_LINE] = LINE_OCCUPIED

    # border safety band
    border_px = int(1.0 / GRID_RES)
    nav[0:border_px, :] = OCCUPIED
    nav[-border_px:, :] = OCCUPIED
    nav[:, 0:border_px] = OCCUPIED
    nav[:, -border_px:] = OCCUPIED

    return nav

def inflate(grid, radius_m):
    """
    Inflate only OCCUPIED cells.
    LINE_OCCUPIED cells stay blocked, but are not inflated.
    """
    out = grid.copy()
    r = int(math.ceil(radius_m / GRID_RES))

    for gy, gx in np.argwhere(grid == OCCUPIED):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if in_bounds(nx, ny) and dx * dx + dy * dy <= r * r:
                    if out[ny, nx] == FREE:
                        out[ny, nx] = INFLATED

    return out

# =========================
# FIXED SLOT LAYOUT
# raw-grid coordinates (NOT display-aligned image coordinates)
# These are approximate Town05 parking slots for your current lot.
# =========================
def make_slot_from_grid(slot_id, gx_center, gy0, gy1, yaw_deg, approach_gy, width_cells=5):
    gx0 = int(round(gx_center - width_cells / 2))
    gx1 = int(round(gx_center + width_cells / 2))

    gx0 = max(0, gx0)
    gx1 = min(GRID_W - 1, gx1)
    gy0 = max(0, min(gy0, gy1))
    gy1 = min(GRID_H - 1, max(gy0, gy1))

    cx_g = 0.5 * (gx0 + gx1)
    cy_g = 0.5 * (gy0 + gy1)

    cx, cy = grid_to_world(cx_g, cy_g, ORIGIN_X, ORIGIN_Y)
    ax, ay = grid_to_world(cx_g, approach_gy, ORIGIN_X, ORIGIN_Y)

    return {
        "id": slot_id,
        "gx0": gx0,
        "gx1": gx1,
        "gy0": gy0,
        "gy1": gy1,
        "cx": cx,
        "cy": cy,
        "approach_x": ax,
        "approach_y": ay,
        "yaw": yaw_deg,
        "occupied": False,
    }

def update_slot_occupancy(sem, slots):
    """
    Improved slot occupancy test:
    - check only an inner patch of the slot, not the full slot rectangle
    - count car pixels inside that patch
    - mark occupied only if enough car cells exist
    """
    for s in slots:
        width = s["gx1"] - s["gx0"] + 1
        height = s["gy1"] - s["gy0"] + 1

        # shrink inward to avoid touching borders / neighboring slots
        mx = max(1, int(round(width * 0.20)))
        my = max(1, int(round(height * 0.20)))

        gx0 = max(0, s["gx0"] + mx)
        gx1 = min(GRID_W - 1, s["gx1"] - mx)
        gy0 = max(0, s["gy0"] + my)
        gy1 = min(GRID_H - 1, s["gy1"] - my)

        if gx1 < gx0 or gy1 < gy0:
            s["occupied"] = True
            continue

        patch = sem[gy0:gy1 + 1, gx0:gx1 + 1]

        # car_mask = (patch == SEM_STATIC_CAR) | (patch == SEM_DYNAMIC_CAR)
        car_mask = (patch == STATIC_CAR) | (patch == DYNAMIC_CAR)
        car_pixels = int(np.sum(car_mask))
        patch_area = patch.size

        # ratio-based test is more robust than "any overlap"
        occ_ratio = car_pixels / max(1, patch_area)

        # tune this threshold if needed
        s["occupied"] = (car_pixels >= 4) and (occ_ratio >= 0.08)

        # optional debug fields
        s["debug_car_pixels"] = car_pixels
        s["debug_occ_ratio"] = occ_ratio

def choose_random_empty_slot(slots):
    free_slots = [s for s in slots if not s["occupied"]]
    if not free_slots:
        return None
    return random.choice(free_slots)


# =========================
# A*
# =========================
def local_clearance_penalty(grid, gx, gy, radius_cells):
    if grid[gy, gx] != FREE:
        return 0.0

    best_d2 = None

    y0 = max(0, gy - radius_cells)
    y1 = min(GRID_H, gy + radius_cells + 1)
    x0 = max(0, gx - radius_cells)
    x1 = min(GRID_W, gx + radius_cells + 1)

    for ny in range(y0, y1):
        for nx in range(x0, x1):
            if grid[ny, nx] in (OCCUPIED, INFLATED):
                d2 = (gx - nx) ** 2 + (gy - ny) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2

    if best_d2 is None:
        return 0.0

    if best_d2 <= 1:
        return 1
    elif best_d2 <= 4:
        return 0.25
    else:
        return 0.0
    
def astar(grid, start, goal):
    moves = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
    ]
    heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_cost = {start: 0.0}
    visited = set()

    while heap:
        _, cg, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)

        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]

        cx, cy = cur
        for dx, dy, step_cost in moves:
            nb = (cx + dx, cy + dy)
            if not in_bounds(*nb):
                continue
            if grid[nb[1], nb[0]] in (OCCUPIED, INFLATED, LINE_OCCUPIED):
                continue

            # tg = cg + step_cost
            clearance_pen = local_clearance_penalty(grid, nb[0], nb[1], radius_cells=3)
            tg = cg + step_cost + clearance_pen
            if tg < g_cost.get(nb, 1e18):
                g_cost[nb] = tg
                came_from[nb] = cur
                heapq.heappush(heap, (tg + heuristic(nb, goal), tg, nb))

    return None


def nearest_free(grid, gx, gy, r=20):
    if in_bounds(gx, gy) and grid[gy, gx] == FREE:
        return (gx, gy)
    for rad in range(1, r + 1):
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                nx, ny = gx + dx, gy + dy
                if in_bounds(nx, ny) and grid[ny, nx] == FREE:
                    return (nx, ny)
    return None


def sparsify(path, step=4):
    if len(path) <= 2:
        return path
    out = [path[0]]
    for i in range(step, len(path), step):
        out.append(path[i])
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out


def plan(grid, start_xy, goal_xy):
    s = world_to_grid(start_xy[0], start_xy[1], ORIGIN_X, ORIGIN_Y)
    g = world_to_grid(goal_xy[0], goal_xy[1], ORIGIN_X, ORIGIN_Y)

    if not in_bounds(*s):
        raise RuntimeError(f"Start {start_xy} outside grid (gx={s[0]}, gy={s[1]})")
    if not in_bounds(*g):
        raise RuntimeError(f"Goal {goal_xy} outside grid (gx={g[0]}, gy={g[1]})")

    sf = nearest_free(grid, *s, 20)
    gf = nearest_free(grid, *g, 20)

    if not sf:
        raise RuntimeError("No free start cell near ego")
    if not gf:
        raise RuntimeError("No free goal cell near target")

    print("[PLAN] running A*...")
    path = astar(grid, sf, gf)
    print("[PLAN] A* done")
    if not path:
        raise RuntimeError("A* found no path")

    world_path = [grid_to_world(gx, gy, ORIGIN_X, ORIGIN_Y) for gx, gy in path]
    return sparsify(world_path, 4)


# =========================
# CONTROLLERS
# =========================
def follow_path(vehicle, path, goal_xy):
    lookahead = 2

    while True:
        t = vehicle.get_transform()
        cx, cy, yaw = t.location.x, t.location.y, t.rotation.yaw

        ci = min(range(len(path)), key=lambda i: math.hypot(path[i][0] - cx, path[i][1] - cy))
        ti = min(ci + lookahead, len(path) - 1)
        wx, wy = path[ti]

        dist_to_goal = math.hypot(goal_xy[0] - cx, goal_xy[1] - cy)
        if dist_to_goal < 0.8:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            break

        heading_err = normalize_angle_deg(math.degrees(math.atan2(wy - cy, wx - cx)) - yaw)
        steer = max(-1.0, min(1.0, heading_err / 40.0))
        spd = get_speed(vehicle)

        if dist_to_goal < 5.0:
            throttle = 0.12 if spd < 1.0 else 0.0
            if spd > 1.5:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3))
                time.sleep(0.05)
                continue
        else:
            throttle = 0.3 if spd < 3.0 else 0.0
            if abs(heading_err) > 30:
                throttle = 0.15

        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))
        time.sleep(0.05)


def yaw_to_unit_vec(yaw_deg):
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)

def compute_pre_entry_point(slot, offset_m=1.0):
    """
    Point just OUTSIDE the selected slot, on the aisle side.

    We compute the slot mouth from the slot rectangle boundary first,
    then move a little farther outward into the aisle.
    This guarantees the pre-entry point is outside the slot, not inside it.
    """
    fx, fy = yaw_to_unit_vec(slot["yaw"])

    # slot rectangle size in world meters
    slot_len_m = (slot["gx1"] - slot["gx0"] + 1) * GRID_RES
    half_len_m = 0.5 * slot_len_m

    # from slot center, go to the aisle-side boundary,
    # then continue a bit farther into the road
    px = slot["cx"] - (half_len_m + offset_m) * fx
    py = slot["cy"] - (half_len_m + offset_m) * fy
    return px, py

def compute_final_parking_target(slot, stop_short_m=0.55):
    """
    Final center target inside the slot, but slightly short of the geometric slot center.
    This prevents the car center from going too deep into the slot.
    """
    fx, fy = yaw_to_unit_vec(slot["yaw"])
    tx = slot["cx"] - stop_short_m * fx
    ty = slot["cy"] - stop_short_m * fy
    return tx, ty

def slot_frame_errors(vehicle, slot, target_xy=None):
    """
    Compute ego error in the slot frame.

    If target_xy is given, use that point as the final stopping target
    instead of the geometric slot center.
    """
    tf = vehicle.get_transform()
    cx, cy, yaw = tf.location.x, tf.location.y, tf.rotation.yaw

    if target_xy is None:
        sx, sy = slot["cx"], slot["cy"]
    else:
        sx, sy = target_xy

    dx = sx - cx
    dy = sy - cy

    fx, fy = yaw_to_unit_vec(slot["yaw"])
    lx, ly = -fy, fx

    forward_err = dx * fx + dy * fy
    lateral_err = dx * lx + dy * ly
    yaw_err = normalize_angle_deg(slot["yaw"] - yaw)

    return forward_err, lateral_err, yaw_err

def clear_selected_slot_corridor(work_grid, slot, extra_front_cells=2, half_width_pad=1):
    """
    Open only the chosen slot interior plus a short corridor from aisle into the slot.
    This lets the final parking maneuver enter the selected slot without globally
    freeing all parking lines.
    """
    gx0 = slot["gx0"]
    gx1 = slot["gx1"]
    gy0 = slot["gy0"]
    gy1 = slot["gy1"]

    # pad sideways a little
    gx0 = max(0, gx0 - half_width_pad)
    gx1 = min(GRID_W - 1, gx1 + half_width_pad)
    gy0 = max(0, gy0 - half_width_pad)
    gy1 = min(GRID_H - 1, gy1 + half_width_pad)

    # open slot interior first
    work_grid[gy0:gy1 + 1, gx0:gx1 + 1] = FREE

    # open a short corridor from aisle side toward slot mouth
    if slot["yaw"] > 0:   # upward-facing slot in your convention
        c0 = max(0, gy0 - extra_front_cells)
        c1 = gy1
        work_grid[c0:c1 + 1, gx0:gx1 + 1] = FREE
    else:                 # downward-facing slot
        c0 = gy0
        c1 = min(GRID_H - 1, gy1 + extra_front_cells)
        work_grid[c0:c1 + 1, gx0:gx1 + 1] = FREE


def align_to_slot_yaw(vehicle, target_yaw, timeout=8.0):
    """
    Low-speed alignment near the aisle approach point.
    We only care about heading here, not exact slot-center position.
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        tf = vehicle.get_transform()
        yaw = tf.rotation.yaw
        sp = get_speed(vehicle)

        yaw_err = normalize_angle_deg(target_yaw - yaw)

        # success: heading good enough and nearly stopped
        if abs(yaw_err) < 6.0 and sp < 0.25:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            return True

        steer = max(-1.0, min(1.0, yaw_err / 20.0))

        # keep alignment slow; allow a little rolling to turn
        if sp > 0.7:
            throttle = 0.0
            brake = 0.25
        else:
            throttle = 0.10
            brake = 0.0

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        ))
        time.sleep(0.05)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    return False

def compute_reachable_pre_entry_point(slot, grid):
    """
    Find a pre-entry point outside the slot that is actually free/reachable in the aisle.
    Try progressively farther points along the outward slot direction, then snap to nearest free.
    """
    fx, fy = yaw_to_unit_vec(slot["yaw"])

    # slot depth in meters from raw grid
    slot_len_m = (slot["gx1"] - slot["gx0"] + 1) * GRID_RES
    half_len_m = 0.5 * slot_len_m

    # try a few outward offsets from the slot mouth
    for extra_m in [0.8, 1.2, 1.6, 2.0, 2.5]:
        px = slot["cx"] - (half_len_m + extra_m) * fx
        py = slot["cy"] - (half_len_m + extra_m) * fy

        gx, gy = world_to_grid(px, py, ORIGIN_X, ORIGIN_Y)
        free_cell = nearest_free(grid, gx, gy, r=8)
        if free_cell is not None:
            wx, wy = grid_to_world(free_cell[0], free_cell[1], ORIGIN_X, ORIGIN_Y)
            return wx, wy

    # fallback: return geometric point if nothing nearby is free
    px = slot["cx"] - (half_len_m + 2.5) * fx
    py = slot["cy"] - (half_len_m + 2.5) * fy
    return px, py


def drive_straight_into_slot(vehicle, slot, timeout=10.0):
    """
    Enter the chosen slot along the slot axis, but stop slightly short of
    the slot center to avoid over-parking.
    """
    deadline = time.time() + timeout
    target_xy = compute_final_parking_target(slot, stop_short_m=0.55)

    while time.time() < deadline:
        tf = vehicle.get_transform()
        sp = get_speed(vehicle)

        forward_err, lateral_err, yaw_err = slot_frame_errors(vehicle, slot, target_xy=target_xy)

        # debug distance to chosen stopping target
        dist = math.hypot(tf.location.x - target_xy[0], tf.location.y - target_xy[1])

        # success
        if abs(forward_err) < 0.45 and abs(lateral_err) < 0.30 and abs(yaw_err) < 8.0 and sp < 0.20:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            return True

        # if yaw drift becomes too large, brake and let it settle
        if abs(yaw_err) > 15.0:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.35))
            time.sleep(0.05)
            continue

        # steering from lateral offset + a bit of yaw correction
        steer_cmd = 0.65 * lateral_err + 0.25 * math.radians(yaw_err)
        steer = max(-0.40, min(0.40, steer_cmd))

        # slow longitudinal approach
        if forward_err > 1.0:
            throttle = 0.10 if sp < 0.55 else 0.0
        elif forward_err > 0.45:
            throttle = 0.05 if sp < 0.30 else 0.0
        elif forward_err > 0.10:
            throttle = 0.025 if sp < 0.18 else 0.0
        else:
            throttle = 0.0

        brake = 0.0
        if forward_err <= 0.0 and sp > 0.12:
            brake = 0.35
        elif throttle == 0.0 and sp > 0.15:
            brake = 0.20

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        ))
        time.sleep(0.05)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    return False


def pull_in_to_slot(vehicle, slot):
    print("[PARK] Stage 1: align with slot yaw")
    ok_align = align_to_slot_yaw(vehicle, slot["yaw"], timeout=8.0)

    tf = vehicle.get_transform()
    print(f"[PARK] After align: pos=({tf.location.x:.2f}, {tf.location.y:.2f}), yaw={tf.rotation.yaw:.2f}")

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    time.sleep(0.4)

    print("[PARK] Stage 2: drive into slot along slot axis")
    ok_enter = drive_straight_into_slot(vehicle, slot, timeout=10.0)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

    tf = vehicle.get_transform()

    # define target_xy BEFORE using it
    target_xy = compute_final_parking_target(slot, stop_short_m=0.55)

    dist = math.hypot(tf.location.x - target_xy[0], tf.location.y - target_xy[1])
    yaw_err = abs(normalize_angle_deg(slot["yaw"] - tf.rotation.yaw))
    forward_err, lateral_err, _ = slot_frame_errors(vehicle, slot, target_xy=target_xy)

    if abs(forward_err) < 0.45 and abs(lateral_err) < 0.30 and yaw_err < 8.0:
        print("[PARK] Parked ✓")
    else:
        print("[PARK] Parking inaccurate")

    print(f"[PARK] Final pose=({tf.location.x:.2f}, {tf.location.y:.2f}) dist={dist:.2f} yaw_err={yaw_err:.2f}")
    print(f"[PARK] Final slot-frame: forward_err={forward_err:.2f}, lateral_err={lateral_err:.2f}")
    print(f"[PARK] align_ok={ok_align}, enter_ok={ok_enter}")

# =========================
# CONNECT TO CARLA
# =========================
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(20.0)

print(f"Loading {MAP_NAME}…")
world = client.load_world(MAP_NAME)
time.sleep(2.0)

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find("vehicle.tesla.model3")

spawn_tf = carla.Transform(
    carla.Location(x=SPAWN_X, y=SPAWN_Y, z=0.5),
    carla.Rotation(yaw=SPAWN_YAW)
)

vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
if vehicle is None:
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2)]:
        spawn_tf.location.x = SPAWN_X + dx
        spawn_tf.location.y = SPAWN_Y + dy
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if vehicle:
            break

if vehicle is None:
    raise RuntimeError("Could not spawn. Adjust SPAWN_X/Y/YAW.")

vehicle.set_autopilot(False)
vehicle.set_simulate_physics(True)
vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
time.sleep(1.0)

# compute inflation margin to match the ego vehicle size
ego_bb = vehicle.bounding_box

# CARLA bbox extents are half-dimensions
ego_half_width_m = ego_bb.extent.y
ego_half_length_m = ego_bb.extent.x

# Inflate obstacles by ego half-width + a small buffer
INFLATION_RADIUS_M = max(0.5, ego_half_width_m + EGO_CLEARANCE_MARGIN_M)

print(f"[EGO] half width = {ego_half_width_m:.2f} m")
print(f"[EGO] half length = {ego_half_length_m:.2f} m")
print(f"[GRID] inflation radius = {INFLATION_RADIUS_M:.2f} m")

t = vehicle.get_transform()
ego_x, ego_y = t.location.x, t.location.y
print(f"Spawned at ({ego_x:.2f}, {ego_y:.2f})")
print(f"Grid origin: ({ORIGIN_X:.2f}, {ORIGIN_Y:.2f}), size: {GRID_W}x{GRID_H} cells")
sgx, sgy = world_to_grid(ego_x, ego_y, ORIGIN_X, ORIGIN_Y)
print(f"Spawn grid cell: gx={sgx}, gy={sgy}  (must be 0–{GRID_W-1} and 0–{GRID_H-1})")
draw_label(world, ego_x, ego_y, "START", carla.Color(0, 255, 0))

# =========================
# BUILD FIXED SLOT SET + INITIAL GRIDS
# =========================
slots_data = []

def _cluster_consecutive(indices, max_gap=1):
    """
    Cluster sorted integer indices into consecutive groups.
    """
    if len(indices) == 0:
        return []

    groups = [[int(indices[0])]]
    for v in indices[1:]:
        v = int(v)
        if v - groups[-1][-1] <= max_gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups

def _build_slots_from_roadlines(semantic_grid):
    """
    Build slots directly from the RAW semantic grid, with no display rotation/flips.

    Raw-grid meaning:
    - gx increases with world x
    - gy increases with world y

    Strategy:
    1. detect horizontal parking-row lines directly in raw grid
    2. detect vertical separator lines directly in raw grid
    3. build left/right slots around each aisle row directly in raw-grid coordinates
    """
    road_mask = (semantic_grid == ROAD_LINE).astype(np.uint8)

    # Restrict to the area where parking slots exist.
    # These bounds are in RAW GRID coordinates, not display coordinates.
    gx_min, gx_max = 15, 105
    gy_min, gy_max = 20, 90

    roi = road_mask[gy_min:gy_max, gx_min:gx_max]

    # ---------------------------------
    # 1) Find aisle center lines in raw grid
    # In raw grid, the long parking-row lines are mostly vertical in the old display,
    # but here we work directly in gx/gy.
    # Since slots are arranged in columns visually now, detect strong y-bands.
    # ---------------------------------
    row_scores = roi.sum(axis=0)   # sum over gy -> strength by gx
    row_thresh = max(8, int(0.45 * row_scores.max()))
    row_candidates = np.where(row_scores >= row_thresh)[0]
    row_groups = _cluster_consecutive(row_candidates, max_gap=2)

    aisle_x_centers = []
    for g in row_groups:
        if len(g) >= 2:
            aisle_x_centers.append(gx_min + int(round(np.mean(g))))

    print("[SLOT] aisle_x_centers(raw):", aisle_x_centers)

    if len(aisle_x_centers) < 3:
        print("[SLOT] Not enough aisle lines detected in raw grid")
        return []

    if len(aisle_x_centers) > 3:
        scored = [(xc, row_scores[xc - gx_min]) for xc in aisle_x_centers]
        scored.sort(key=lambda t: t[1], reverse=True)
        aisle_x_centers = sorted([t[0] for t in scored[:3]])

    # ---------------------------------
    # 2) Find slot separator lines in raw grid
    # Remove aisle columns first so projection is dominated by separators.
    # ---------------------------------
    sep_only = roi.copy()

    for xc in aisle_x_centers:
        cc = xc - gx_min
        c0 = max(0, cc - 2)
        c1 = min(sep_only.shape[1], cc + 3)
        sep_only[:, c0:c1] = 0

    col_scores = sep_only.sum(axis=1)   # sum over gx -> strength by gy
    col_thresh = max(4, int(0.30 * col_scores.max()))
    col_candidates = np.where(col_scores >= col_thresh)[0]
    col_groups = _cluster_consecutive(col_candidates, max_gap=1)

    y_centers = []
    for g in col_groups:
        if len(g) >= 1:
            y_centers.append(gy_min + int(round(np.mean(g))))

    print("[SLOT] y_centers(raw):", y_centers)

    if len(y_centers) < 2:
        print("[SLOT] Not enough separator rows detected in raw grid")
        return []

    # ---------------------------------
    # 3) Build slots to left/right of each aisle line
    # ---------------------------------
    slots = []
    slot_id = 0

    for aisle_x in aisle_x_centers:
        # separator rows above and below this aisle region
        upper_rows = np.where(sep_only[:, :aisle_x - gx_min].sum(axis=1) > 0)[0]
        lower_rows = np.where(sep_only[:, aisle_x - gx_min + 1:].sum(axis=1) > 0)[0]

        upper_groups = _cluster_consecutive(upper_rows, max_gap=1)
        lower_groups = _cluster_consecutive(lower_rows, max_gap=1)

        if not upper_groups or not lower_groups:
            continue

        # But slot boundaries are really defined by neighboring y separator groups.
        # We use all y centers as slot boundaries and build one slot per gap.
        for i in range(len(y_centers) - 1):
            y0 = y_centers[i]
            y1 = y_centers[i + 1]
            gap = y1 - y0

            if gap < 4 or gap > 9:
                continue

            # left slot of aisle
            gx0 = max(0, aisle_x - 10)
            gx1 = max(0, aisle_x - 2)
            gy0 = max(0, y0 + 1)
            gy1 = min(GRID_H - 1, y1 - 1)

            if gx1 > gx0 and gy1 > gy0:
                cx_g = 0.5 * (gx0 + gx1)
                cy_g = 0.5 * (gy0 + gy1)
                cx, cy = grid_to_world(cx_g, cy_g, ORIGIN_X, ORIGIN_Y)

                agx = aisle_x - 5
                agy = cy_g
                ax, ay = grid_to_world(agx, agy, ORIGIN_X, ORIGIN_Y)

                slots.append({
                    "id": slot_id,
                    "gx0": gx0,
                    "gx1": gx1,
                    "gy0": gy0,
                    "gy1": gy1,
                    "cx": cx,
                    "cy": cy,
                    "approach_x": ax,
                    "approach_y": ay,
                    "yaw": 0.0,
                    "occupied": False,
                })
                slot_id += 1

            # right slot of aisle
            gx0 = min(GRID_W - 1, aisle_x + 2)
            gx1 = min(GRID_W - 1, aisle_x + 10)
            gy0 = max(0, y0 + 1)
            gy1 = min(GRID_H - 1, y1 - 1)

            if gx1 > gx0 and gy1 > gy0:
                cx_g = 0.5 * (gx0 + gx1)
                cy_g = 0.5 * (gy0 + gy1)
                cx, cy = grid_to_world(cx_g, cy_g, ORIGIN_X, ORIGIN_Y)

                agx = aisle_x + 5
                agy = cy_g
                ax, ay = grid_to_world(agx, agy, ORIGIN_X, ORIGIN_Y)

                slots.append({
                    "id": slot_id,
                    "gx0": gx0,
                    "gx1": gx1,
                    "gy0": gy0,
                    "gy1": gy1,
                    "cx": cx,
                    "cy": cy,
                    "approach_x": ax,
                    "approach_y": ay,
                    "yaw": 180.0,
                    "occupied": False,
                })
                slot_id += 1

    print(f"[SLOT] slots built from RAW road lines: {len(slots)}")
    return slots

def refresh_semantics_and_nav():
    global semantic_grid, nav_grid, inflated_grid, slots_data, road_line_bbs_cache

    print("[GRID] Building semantic grid from occupancy_grid.py ...")
    result = collect_lot_semantics(world, include_dynamic=True, ego_actor_id=vehicle.id)

    semantic_grid = result["grid"]
    road_line_bbs_cache = result["road_line_bbs"]

    slots_data = _build_slots_from_roadlines(semantic_grid)

    update_slot_occupancy(semantic_grid, slots_data)
    n_occ = sum(1 for s in slots_data if s["occupied"])
    n_free = sum(1 for s in slots_data if not s["occupied"])
    print(f"[SLOT] total={len(slots_data)} free={n_free} occupied={n_occ}")

    nav_grid = build_nav_grid_from_semantics(semantic_grid)
    inflated_grid = inflate(nav_grid, INFLATION_RADIUS_M)
    print(f"[GRID] Inflated occupied cells: {int(np.sum(inflated_grid != FREE))}")
    
refresh_semantics_and_nav()

# =========================
# CAMERA
# =========================
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(CAM_IMG_W))
cam_bp.set_attribute("image_size_y", str(CAM_IMG_H))
cam_bp.set_attribute("fov", str(CAM_HFOV_DEG))

camera = world.spawn_actor(
    cam_bp,
    carla.Transform(
        carla.Location(x=CAMERA_X, y=CAMERA_Y, z=CAMERA_Z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )
)


def draw_all_empty_slots_overlay(frame, slots, color=(255, 0, 0), alpha=0.22):
    """
    Draw all currently detected empty slots in blue.
    OpenCV uses BGR, so (255, 0, 0) = blue.
    """
    overlay = frame.copy()

    for slot in slots:
        if slot["occupied"]:
            continue

        wx0, wy0 = grid_to_world(slot["gx0"], slot["gy0"], ORIGIN_X, ORIGIN_Y)
        wx1, wy1 = grid_to_world(slot["gx1"], slot["gy1"], ORIGIN_X, ORIGIN_Y)

        px0, py0 = world_to_pixel(wx0, wy0)
        px1, py1 = world_to_pixel(wx1, wy1)

        x0, x1 = sorted([px0, px1])
        y0, y1 = sorted([py0, py1])

        x0 = max(0, min(CAM_IMG_W - 1, x0))
        x1 = max(0, min(CAM_IMG_W - 1, x1))
        y0 = max(0, min(CAM_IMG_H - 1, y0))
        y1 = max(0, min(CAM_IMG_H - 1, y1))

        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=1)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def world_to_pixel(wx, wy):
    px = int((wy - CAMERA_Y) / CAM_SPAN_Y * CAM_IMG_W + CAM_IMG_W / 2)
    py = int((CAMERA_X - wx) / CAM_SPAN_X * CAM_IMG_H + CAM_IMG_H / 2)
    return px, py

def draw_slot_overlay(frame, slot, color=(0, 255, 0), alpha=0.35):
    overlay = frame.copy()

    wx0, wy0 = grid_to_world(slot["gx0"], slot["gy0"], ORIGIN_X, ORIGIN_Y)
    wx1, wy1 = grid_to_world(slot["gx1"], slot["gy1"], ORIGIN_X, ORIGIN_Y)

    px0, py0 = world_to_pixel(wx0, wy0)
    px1, py1 = world_to_pixel(wx1, wy1)

    x0, x1 = sorted([px0, px1])
    y0, y1 = sorted([py0, py1])

    x0 = max(0, min(CAM_IMG_W - 1, x0))
    x1 = max(0, min(CAM_IMG_W - 1, x1))
    y0 = max(0, min(CAM_IMG_H - 1, y0))
    y1 = max(0, min(CAM_IMG_H - 1, y1))

    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=2)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def draw_cached_road_lines_overlay(frame):
    if not road_line_bbs_cache:
        return frame

    overlay = frame.copy()

    for bb in road_line_bbs_cache:
        try:
            w = bb.extent.x * 2.0
            h = bb.extent.y * 2.0

            if h > w:
                color = (255, 255, 0)   # cyan-ish
            else:
                color = (0, 255, 255)   # yellow

            verts = bb.get_world_vertices(carla.Transform())
            pts = []

            # project all 8 vertices, then draw bounding rectangle in image space
            for v in verts:
                px, py = world_to_pixel(v.x, v.y)
                pts.append([px, py])

            pts = np.array(pts, dtype=np.int32)

            x, y, w_box, h_box = cv2.boundingRect(pts)
            cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), color, 2)

        except Exception as e:
            print("[DRAW] road-line bbox failed:", e)

    return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

def draw_ego_vehicle_overlay(frame, vehicle, color=(0, 0, 255), thickness=2):
    """
    Draw ego vehicle from real rotated CARLA bbox, not from grid cells.
    This fixes the display mismatch.
    """
    bb = vehicle.bounding_box
    tf = vehicle.get_transform()
    verts = bb.get_world_vertices(tf)

    pts = []
    for v in verts:
        px, py = world_to_pixel(v.x, v.y)
        pts.append([px, py])

    pts = np.array(pts, dtype=np.int32)
    x, y, w_box, h_box = cv2.boundingRect(pts)

    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, thickness)
    return frame

def draw_slot_axis(frame, slot, length=20, color=(0, 255, 0), thickness=2):
    fx, fy = yaw_to_unit_vec(slot["yaw"])

    x0, y0 = slot["cx"], slot["cy"]
    x1 = x0 + fx * 3.0
    y1 = y0 + fy * 3.0

    p0 = world_to_pixel(x0, y0)
    p1 = world_to_pixel(x1, y1)

    cv2.line(frame, p0, p1, color, thickness)
    return frame

def process_image(image):
    frame = np.frombuffer(image.raw_data, dtype=np.uint8)
    frame = frame.reshape((image.height, image.width, 4))[:, :, :3].copy()

    # ---------------------------------
    # Overlay nav-grid debug
    # red    = OCCUPIED
    # orange = INFLATED
    # ---------------------------------
    if inflated_grid is not None:
        overlay = frame.copy()

        for gy in range(GRID_H):
            for gx in range(GRID_W):
                cell = inflated_grid[gy, gx]
                if cell == FREE:
                    continue

                wx, wy = grid_to_world(gx, gy, ORIGIN_X, ORIGIN_Y)
                px, py = world_to_pixel(wx, wy)

                if 0 <= px < CAM_IMG_W and 0 <= py < CAM_IMG_H:
                    if cell == OCCUPIED:
                        color = (0, 0, 255)        # red
                    elif cell == INFLATED:
                        color = (0, 165, 255)      # orange
                    elif cell == LINE_OCCUPIED:
                        color = (0, 255, 255)      # yellow
                    else:
                        continue

                    cv2.rectangle(
                        overlay,
                        (px - 2, py - 2),
                        (px + 2, py + 2),
                        color,
                        thickness=-1
                    )

        alpha = 0.35
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # ---------------------------------
        # Draw all currently empty slots (BLUE)
        # ---------------------------------
        if slots_data:
            frame = draw_all_empty_slots_overlay(frame, slots_data, color=(255, 0, 0), alpha=0.18)

        # ---------------------------------
        # Draw selected slot (GREEN)
        # ---------------------------------
        slot = selected_slot["slot"]
        if slot is not None:
            frame = draw_slot_overlay(frame, slot, color=(0, 255, 0), alpha=0.30)
            # draw yaw direction after delected the slot
            frame = draw_slot_axis(frame, slot, color=(0, 255, 0), thickness=2)


            

            px, py = world_to_pixel(slot["cx"], slot["cy"])
            cv2.drawMarker(frame, (px, py), (0, 255, 0),
                        cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)

            px_pre = slot.get("pre_entry_x", slot["approach_x"])
            py_pre = slot.get("pre_entry_y", slot["approach_y"])
            apx, apy = world_to_pixel(px_pre, py_pre)
            cv2.circle(frame, (apx, apy), 8, (255, 0, 255), 2)   # pink
          
        
    # ---------------------------------
    # Draw chosen slot marker
    # ---------------------------------
    slot = selected_slot["slot"]
    if slot is not None:
        px, py = world_to_pixel(slot["cx"], slot["cy"])
        cv2.drawMarker(frame, (px, py), (0, 0, 255),
                       cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)

        apx, apy = world_to_pixel(slot["approach_x"], slot["approach_y"])
        cv2.circle(frame, (apx, apy), 8, (255, 255, 0), 2)

    # ---------------------------------
    # Draw ego vehicle marker
    # ---------------------------------
    frame = draw_ego_vehicle_overlay(frame, vehicle, color=(0, 0, 255), thickness=2)
    

    if not image_queue.empty():
        try:
            image_queue.get_nowait()
        except queue.Empty:
            pass
    image_queue.put(frame)

    
camera.listen(process_image)

# =========================
# PARKING WORKER
# =========================
# def is_vehicle_clear_of_slot(vehicle, slot, margin_m=0.8):
#     """
#     True when ego center has reversed past the slot mouth.
#     """
#     tf = vehicle.get_transform()
#     cx, cy = tf.location.x, tf.location.y

#     fx, fy = yaw_to_unit_vec(slot["yaw"])
#     dx = slot["cx"] - cx
#     dy = slot["cy"] - cy

#     forward_err = dx * fx + dy * fy

#     slot_len_m = (slot["gx1"] - slot["gx0"] + 1) * GRID_RES
#     half_len_m = 0.5 * slot_len_m

#     return forward_err >= half_len_m + margin_m
def is_vehicle_clear_of_slot(vehicle, slot, margin_m=0.8):
    """
    True when ego center is outside the slot rectangle plus margin.
    Uses both longitudinal and lateral slot-frame distance, so it works
    even if yaw sign/direction convention is imperfect.
    """
    tf = vehicle.get_transform()
    cx, cy = tf.location.x, tf.location.y

    fx, fy = yaw_to_unit_vec(slot["yaw"])
    lx, ly = -fy, fx

    dx = cx - slot["cx"]
    dy = cy - slot["cy"]

    longitudinal = dx * fx + dy * fy
    lateral = dx * lx + dy * ly

    slot_len_m = (slot["gx1"] - slot["gx0"] + 1) * GRID_RES
    slot_wid_m = (slot["gy1"] - slot["gy0"] + 1) * GRID_RES

    half_len_m = 0.5 * slot_len_m
    half_wid_m = 0.5 * slot_wid_m

    # clear if ego center is outside the slot box by a small margin
    return (
        abs(longitudinal) > half_len_m + margin_m
        or abs(lateral) > half_wid_m + margin_m
    )


def reverse_until_clear_of_slot(vehicle, slot, timeout=14.0):
    """
    Stage 1 of leave:
    reverse until ego clears the slot corridor.
    """
    deadline = time.time() + timeout

    # nominal reverse target: use stored pre-entry if available
    target_xy = (
        slot.get("pre_entry_x", slot["approach_x"]),
        slot.get("pre_entry_y", slot["approach_y"]),
    )

    while time.time() < deadline:
        if is_vehicle_clear_of_slot(vehicle, slot, margin_m=0.8):
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, reverse=False
            ))
            print("[LEAVE] Slot cleared")
            return True

        tf = vehicle.get_transform()
        cx, cy, yaw = tf.location.x, tf.location.y, tf.rotation.yaw
        spd = get_speed(vehicle)

        desired_heading = math.degrees(math.atan2(target_xy[1] - cy, target_xy[0] - cx))
        heading_err = normalize_angle_deg(desired_heading - yaw)

        # reverse steering uses opposite sign
        steer = max(-0.8, min(0.8, -heading_err / 35.0))

        throttle = 0.18 if spd < 1.0 else 0.0
        brake = 0.0 if spd < 1.2 else 0.25

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            reverse=True
        ))
        time.sleep(0.05)

    vehicle.apply_control(carla.VehicleControl(
        throttle=0.0, steer=0.0, brake=1.0, reverse=False
    ))

    print("[LEAVE] Reverse timeout")
    return is_vehicle_clear_of_slot(vehicle, slot, margin_m=0.8)


def run_leave():
    if not run_lock.acquire(blocking=False):
        demo_state["message"] = "Already running"
        return

    try:
        slot = last_parked_slot["slot"]
        if slot is None:
            demo_state["phase"] = "error"
            demo_state["message"] = "No parked slot to leave from"
            return

        demo_state["phase"] = "running"
        demo_state["message"] = "Leaving: reversing out of slot…"
        demo_state["slot_id"] = slot["id"]

        selected_slot["slot"] = slot

        # Stage 1: pull out completely
        cleared = reverse_until_clear_of_slot(vehicle, slot, timeout=14.0)
        if not cleared:
            demo_state["phase"] = "error"
            demo_state["message"] = "Could not clear slot"
            return

        vehicle.apply_control(carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=1.0, reverse=False
        ))
        time.sleep(0.5)

        # Stage 2: A* back to spawn
        demo_state["message"] = "Leaving: planning path back to spawn…"

        refresh_semantics_and_nav()
        work_grid = inflated_grid.copy()
        # Re-open the old slot corridor and aisle mouth so A* can escape
        clear_selected_slot_corridor(
            work_grid,
            slot,
            extra_front_cells=6,
            half_width_pad=2
        )

        # Open around current ego pose
        tf = vehicle.get_transform()
        start_xy = (tf.location.x, tf.location.y)

        sgx, sgy = world_to_grid(start_xy[0], start_xy[1], ORIGIN_X, ORIGIN_Y)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                if in_bounds(sgx + dx, sgy + dy):
                    work_grid[sgy + dy, sgx + dx] = FREE

        # Open around pre-entry / aisle point too
        pre_x = slot.get("pre_entry_x", slot["approach_x"])
        pre_y = slot.get("pre_entry_y", slot["approach_y"])
        pgx, pgy = world_to_grid(pre_x, pre_y, ORIGIN_X, ORIGIN_Y)

        for dy in range(-6, 7):
            for dx in range(-6, 7):
                if in_bounds(pgx + dx, pgy + dy):
                    work_grid[pgy + dy, pgx + dx] = FREE

        # Open around spawn goal
        spawn_goal = (SPAWN_X, SPAWN_Y)
        ggx, ggy = world_to_grid(spawn_goal[0], spawn_goal[1], ORIGIN_X, ORIGIN_Y)

        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if in_bounds(ggx + dx, ggy + dy):
                    work_grid[ggy + dy, ggx + dx] = FREE

        tf = vehicle.get_transform()
        start_xy = (tf.location.x, tf.location.y)
        spawn_goal = (SPAWN_X, SPAWN_Y)

        # open ego start area
        sgx, sgy = world_to_grid(start_xy[0], start_xy[1], ORIGIN_X, ORIGIN_Y)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if in_bounds(sgx + dx, sgy + dy):
                    work_grid[sgy + dy, sgx + dx] = FREE

        # open spawn goal area
        ggx, ggy = world_to_grid(spawn_goal[0], spawn_goal[1], ORIGIN_X, ORIGIN_Y)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if in_bounds(ggx + dx, ggy + dy):
                    work_grid[ggy + dy, ggx + dx] = FREE

        path = plan(work_grid, start_xy, spawn_goal)
        draw_path(world, path, carla.Color(255, 120, 0))

        demo_state["message"] = "Leaving: driving back to spawn…"
        follow_path(vehicle, path, spawn_goal)

        vehicle.apply_control(carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=1.0, reverse=False
        ))

        selected_slot["slot"] = None
        last_parked_slot["slot"] = None

        demo_state["phase"] = "done"
        demo_state["message"] = "Returned to spawn"
        demo_state["slot_id"] = None
        demo_state["is_parked"] = False

    except Exception as e:
        demo_state["phase"] = "error"
        demo_state["message"] = str(e)
        print("[ERROR leave]", e)
    finally:
        run_lock.release()

def run_random_park():
    if not run_lock.acquire(blocking=False):
        demo_state["message"] = "Already running"
        return

    try:
        demo_state["phase"] = "running"
        demo_state["message"] = "Searching for random empty slot…"
        demo_state["slot_id"] = None

        refresh_semantics_and_nav()

        slot = choose_random_empty_slot(slots_data)
        if slot is None:
            demo_state["phase"] = "error"
            demo_state["message"] = "No empty slot available"
            print("[SLOT] No empty slot available")
            return

        selected_slot["slot"] = slot
        
        pre_x, pre_y = compute_reachable_pre_entry_point(slot, inflated_grid)
        slot["pre_entry_x"] = pre_x
        slot["pre_entry_y"] = pre_y

        print(f"[PARK] pre-entry = ({pre_x:.2f}, {pre_y:.2f})")
        print(f"[PARK] slot center = ({slot['cx']:.2f}, {slot['cy']:.2f})")

        demo_state["slot_id"] = slot["id"]
        demo_state["message"] = (
            f"Chosen slot #{slot['id']} at ({slot['cx']:.2f}, {slot['cy']:.2f}), "
            f"yaw={slot['yaw']:.1f}"
        )

        print(f"[SLOT] Chosen slot #{slot['id']} center=({slot['cx']:.2f}, {slot['cy']:.2f}) yaw={slot['yaw']:.1f}")
        draw_label(world, slot["cx"], slot["cy"], f"SLOT {slot['id']}", carla.Color(255, 0, 0))
        draw_label(world, slot["approach_x"], slot["approach_y"], "APPROACH", carla.Color(0, 255, 255))

        # Open only the selected slot corridor + small holes at approach / center
        work_grid = inflated_grid.copy()

        clear_selected_slot_corridor(
            work_grid,
            slot,
            extra_front_cells=3,
            half_width_pad=1
        )

       
        final_tx, final_ty = compute_final_parking_target(slot, stop_short_m=0.55)

        for wx, wy in [
            (slot["pre_entry_x"], slot["pre_entry_y"]),
            (final_tx, final_ty),
        ]:
            gx, gy = world_to_grid(wx, wy, ORIGIN_X, ORIGIN_Y)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if in_bounds(gx + dx, gy + dy):
                        work_grid[gy + dy, gx + dx] = FREE
                        
        tf = vehicle.get_transform()
        start_xy = (tf.location.x, tf.location.y)

        # Punch a hole around ego start so nearest_free can succeed
        sgx, sgy = world_to_grid(start_xy[0], start_xy[1], ORIGIN_X, ORIGIN_Y)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if in_bounds(sgx + dx, sgy + dy):
                    work_grid[sgy + dy, sgx + dx] = FREE

        demo_state["message"] = "Planning path…"
        path = plan(work_grid, start_xy, (slot["pre_entry_x"], slot["pre_entry_y"]))
        draw_path(world, path, carla.Color(0, 200, 255))

        demo_state["message"] = "Driving to slot approach…"
        time.sleep(1.0)
        follow_path(vehicle, path, (slot["pre_entry_x"], slot["pre_entry_y"]))
       
        demo_state["message"] = "Pulling into slot…"
        pull_in_to_slot(vehicle, slot)

        tf = vehicle.get_transform()
        dist = math.hypot(tf.location.x - slot["cx"], tf.location.y - slot["cy"])
        yaw_err = abs(normalize_angle_deg(slot["yaw"] - tf.rotation.yaw))
        print(f"[PARK] Final pose=({tf.location.x:.2f}, {tf.location.y:.2f}) dist={dist:.2f} yaw_err={yaw_err:.2f}")

        last_parked_slot["slot"] = dict(slot)
        demo_state["is_parked"] = True

        # after 1s, remove the parking helper markers
        time.sleep(1.0)
        selected_slot["slot"] = None
        demo_state["phase"] = "done"
        demo_state["message"] = f"Parked in slot #{slot['id']}"
    except Exception as e:
        demo_state["phase"] = "error"
        demo_state["message"] = str(e)
        print("[ERROR]", e)
    finally:
        run_lock.release()


# =========================
# FLASK APP
# =========================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>CARLA Parking</title>
<style>
  body {
    background:#111;
    color:#eee;
    font-family:sans-serif;
    display:flex;
    flex-direction:column;
    align-items:center;
    margin:0;
    padding:20px;
  }
  h2 { margin: 10px 0 8px; }
  #info { font-size:14px; margin-bottom:10px; color:#8ef; }
  #wrap { position:relative; }
  #wrap img { display:block; border:1px solid #333; }
  #btn, #leaveBtn {
    margin-top:12px;
    padding:10px 24px;
    font-size:16px;
    background:#1a8;
    color:#fff;
    border:none;
    border-radius:6px;
    cursor:pointer;
  }
  #btn:hover, #leaveBtn:hover { background:#1ca; }
  #btn:disabled, #leaveBtn:disabled { background:#666; cursor:not-allowed; }
  #status {
    margin-top:10px;
    font-size:14px;
    color:#fa8;
  }
</style>
</head>
<body>
<h2>CARLA Parking Demo</h2>
<div id="info">Press the button to choose a random empty slot and park there</div>

<div id="wrap">
  <img src="/video_feed" width="1000" id="feed">
</div>

<div style="display:flex; gap:12px; margin-top:12px;">
  <button id="btn" onclick="parkRandom()">🎲 Park in Random Empty Slot</button>
  <button id="leaveBtn" onclick="leaveSlot()" disabled>↩️ Leave</button>
</div>
<div id="status">Ready</div>

<script>
function parkRandom() {
  const btn = document.getElementById('btn');
  btn.disabled = true;

  fetch('/park_random', { method:'POST' })
    .then(r => r.json())
    .then(d => {
      document.getElementById('status').textContent = d.message;
      pollStatus();
    })
    .catch(err => {
      document.getElementById('status').textContent = 'Request failed';
      btn.disabled = false;
    });
}

function leaveSlot() {
  const btn = document.getElementById('btn');
  const leaveBtn = document.getElementById('leaveBtn');
  btn.disabled = true;
  leaveBtn.disabled = true;

  fetch('/leave', { method:'POST' })
    .then(r => r.json())
    .then(d => {
      document.getElementById('status').textContent = d.message;
      pollStatus();
    })
    .catch(err => {
      document.getElementById('status').textContent = 'Leave request failed';
      btn.disabled = false;
      leaveBtn.disabled = false;
    });
}

function pollStatus() {
  fetch('/status')
    .then(r => r.json())
    .then(d => {
      let msg = 'Phase: ' + d.phase;
      if (d.slot_id !== null) {
        msg += ' | slot #' + d.slot_id;
      }
      if (d.message) {
        msg += ' | ' + d.message;
      }
      document.getElementById('status').textContent = msg;
      
      const leaveBtn = document.getElementById('leaveBtn');
      if (d.is_parked && d.phase !== 'running') {
        leaveBtn.disabled = false;
      } else {
        leaveBtn.disabled = true;
      }

      if (d.phase === 'running' || d.phase === 'waiting') {
        setTimeout(pollStatus, 1000);
      } else {
        document.getElementById('btn').disabled = false;
      }
    })
    .catch(err => {
      document.getElementById('status').textContent = 'Status polling failed';
      document.getElementById('btn').disabled = false;
    });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

@app.route("/park_random", methods=["POST"])
def park_random():
    if demo_state["phase"] == "running":
        return jsonify({"message": "Already running"}), 400

    demo_state["phase"] = "waiting"
    demo_state["message"] = "Starting random parking job…"
    demo_state["slot_id"] = None

    thread = threading.Thread(target=run_random_park, daemon=True)
    thread.start()

    return jsonify({"message": "Random empty slot requested. Starting now…"})

@app.route("/leave", methods=["POST"])
def leave():
    if demo_state["phase"] == "running":
        return jsonify({"message": "Already running"}), 400

    if not demo_state.get("is_parked", False):
        return jsonify({"message": "Vehicle is not parked"}), 400

    demo_state["phase"] = "waiting"
    demo_state["message"] = "Starting leave job…"

    thread = threading.Thread(target=run_leave, daemon=True)
    thread.start()

    return jsonify({"message": "Leave requested. Starting now…"})

@app.route("/status")
def status():
    return jsonify(demo_state)

def gen_frames():
    while True:
        frame = image_queue.get()
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# RUN
# =========================
try:
    print(f"Open http://{STREAM_HOST}:{STREAM_PORT} then press the random-park button")
    app.run(host=STREAM_HOST, port=STREAM_PORT, threaded=True)
finally:
    for actor in [camera, vehicle]:
        try:
            actor.stop()
        except Exception:
            pass
        try:
            actor.destroy()
        except Exception:
            pass
