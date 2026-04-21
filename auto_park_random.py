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
INFLATION_RADIUS_M = 0.5

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
SPAWN_Y   = -48.03
SPAWN_YAW =  90.0

# =========================
# SHARED STATE
# =========================
demo_state = {
    "phase": "waiting",
    "message": "Ready",
    "slot_id": None,
}
selected_slot = {"slot": None}
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


# =========================
# SEMANTIC GRID BUILD
# =========================
# def _mark_semantic_drivable(world, sem_grid):
#     carla_map = world.get_map()
#     lane_mask = carla.LaneType.Driving | carla.LaneType.Parking

#     for gy in range(GRID_H):
#         for gx in range(GRID_W):
#             wx, wy = grid_to_world(gx, gy, ORIGIN_X, ORIGIN_Y)
#             loc = carla.Location(x=wx, y=wy, z=0.5)
#             wp = carla_map.get_waypoint(loc, project_to_road=False, lane_type=lane_mask)
#             if wp is not None:
#                 sem_grid[gy, gx] = SEM_DRIVABLE


# def build_semantic_grid(world, ego_vehicle):
#     sem = np.zeros((GRID_H, GRID_W), dtype=np.uint8)

#     # 1) drivable / parking cells
#     _mark_semantic_drivable(world, sem)

#     # 2) static map vehicles
#     static_labels = [
#         carla.CityObjectLabel.Car,
#         carla.CityObjectLabel.Truck,
#         carla.CityObjectLabel.Bus,
#         carla.CityObjectLabel.Motorcycle,
#         carla.CityObjectLabel.Bicycle,
#     ]
#     for label in static_labels:
#         try:
#             bbs = world.get_level_bbs(label)
#             bbs = [bb for bb in bbs if bbox_center_in_lot(bb)]
#             for bb in bbs:
#                 _rasterize_bbox_semantic(sem, bb.location, bb.extent, SEM_STATIC_CAR)
#         except Exception as e:
#             print(f"[SEM] Could not query {label}: {e}")

#     # 3) dynamic vehicles except ego
#     for actor in world.get_actors().filter("vehicle.*"):
#         if actor.id == ego_vehicle.id:
#             continue
#         at = actor.get_transform()
#         bb = actor.bounding_box
#         draw_rotated_box(
#             sem,
#             at.location.x,
#             at.location.y,
#             math.radians(at.rotation.yaw),
#             bb.extent.x + 0.2,
#             bb.extent.y + 0.2,
#             ORIGIN_X,
#             ORIGIN_Y,
#             SEM_DYNAMIC_CAR
#         )

#     try:
#         global road_line_bbs_cache

#         road_line_bbs = world.get_level_bbs(carla.CityObjectLabel.RoadLines)
#         road_line_bbs = [bb for bb in road_line_bbs if bbox_center_in_lot(bb)]
#         road_line_bbs_cache = road_line_bbs

#         print(f"[SEM] Road-line bboxes in lot: {len(road_line_bbs)}")
#         for bb in road_line_bbs:
#             _rasterize_bbox_semantic(
#                 sem,
#                 bb.location,
#                 bb.extent,
#                 SEM_ROAD_LINE,
#                 shrink_x=0.12,
#                 shrink_y=0.12
#             )
#     except Exception as e:
#         print(f"[SEM] Could not query road lines: {e}")

#     return sem


def build_nav_grid_from_semantics(sem):
    """
    Navigation grid:
    - lot interior starts FREE
    - cars are OCCUPIED (and later inflated)
    - parking lines are LINE_OCCUPIED (blocked, but not inflated)
    """
    nav = np.full((GRID_H, GRID_W), FREE, dtype=np.uint8)

    # # hard obstacles that should be inflated
    # nav[sem == SEM_STATIC_CAR] = OCCUPIED
    # nav[sem == SEM_DYNAMIC_CAR] = OCCUPIED

    # # painted parking lines: blocked, but no inflation
    # nav[sem == SEM_ROAD_LINE] = LINE_OCCUPIED

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


def display_grid_to_raw(dgx, dgy):
    """
    Convert display-aligned occupancy-grid coordinates back to raw-grid coordinates.

    occupancy_grid.py display uses:
        grid_vis = np.rot90(grid, k=1)
        grid_vis = np.flipud(grid_vis)

    For this square grid, that is effectively a transpose:
        raw_gx = display_gy
        raw_gy = display_gx
    """
    raw_gx = int(round(dgy))
    raw_gy = int(round(dgx))
    return raw_gx, raw_gy


def make_slot_from_display_grid(slot_id, dgx_center, dgy0, dgy1, yaw_deg, approach_dgy, width_cells=5):
    """
    Create a slot using coordinates measured from the DISPLAY-ALIGNED occupancy image,
    then convert them back into raw-grid coordinates for planning.
    """
    dgx0 = int(round(dgx_center - width_cells / 2))
    dgx1 = int(round(dgx_center + width_cells / 2))

    # Convert display-rect corners to raw-grid corners by swapping axes
    gx0, gy0 = display_grid_to_raw(dgx0, dgy0)
    gx1, gy1 = display_grid_to_raw(dgx1, dgy1)

    gx0, gx1 = sorted([gx0, gx1])
    gy0, gy1 = sorted([gy0, gy1])

    gx0 = max(0, min(GRID_W - 1, gx0))
    gx1 = max(0, min(GRID_W - 1, gx1))
    gy0 = max(0, min(GRID_H - 1, gy0))
    gy1 = max(0, min(GRID_H - 1, gy1))

    cx_g = 0.5 * (gx0 + gx1)
    cy_g = 0.5 * (gy0 + gy1)

    cx, cy = grid_to_world(cx_g, cy_g, ORIGIN_X, ORIGIN_Y)

    agx, agy = display_grid_to_raw(dgx_center, approach_dgy)
    ax, ay = grid_to_world(agx, agy, ORIGIN_X, ORIGIN_Y)

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

def build_fixed_slots():
    """
    Fixed slot geometry measured from the DISPLAY-ALIGNED occupancy image.
    We convert those coordinates back into raw-grid coordinates before planning.
    """
    slots = []
    slot_id = 0

    # These were estimated from your display-aligned occupancy screenshot
    display_x_centers = [28, 33, 39, 46, 52, 58, 64, 70, 76, 80]

    # (aisle_dgy, lower_slot_dgy0, lower_slot_dgy1, upper_slot_dgy0, upper_slot_dgy1)
    # These are in DISPLAY-GRID coordinates
    row_specs = [
        (28, 17, 26, 30, 39),
        (60, 49, 58, 62, 70),
        (93, 82, 91, 95, 105),
    ]

    for aisle_dgy, lower0, lower1, upper0, upper1 in row_specs:
        for dgx in display_x_centers:
            # lower slot in display image
            slots.append(
                make_slot_from_display_grid(
                    slot_id=slot_id,
                    dgx_center=dgx,
                    dgy0=lower0,
                    dgy1=lower1,
                    yaw_deg=-90.0,
                    approach_dgy=aisle_dgy - 3,
                    width_cells=5,
                )
            )
            slot_id += 1

            # upper slot in display image
            slots.append(
                make_slot_from_display_grid(
                    slot_id=slot_id,
                    dgx_center=dgx,
                    dgy0=upper0,
                    dgy1=upper1,
                    yaw_deg=90.0,
                    approach_dgy=aisle_dgy + 3,
                    width_cells=5,
                )
            )
            slot_id += 1

    print(f"[SLOT] fixed slots created: {len(slots)}")
    return slots

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

            tg = cg + step_cost
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

    path = astar(grid, sf, gf)
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


def pull_in_to_slot(vehicle, slot):
    slot_x = slot["cx"]
    slot_y = slot["cy"]
    target_yaw = slot["yaw"]

    deadline = time.time() + 12.0

    while time.time() < deadline:
        t = vehicle.get_transform()
        cx, cy, yaw = t.location.x, t.location.y, t.rotation.yaw

        dist = math.hypot(slot_x - cx, slot_y - cy)
        yaw_err = normalize_angle_deg(target_yaw - yaw)
        sp = get_speed(vehicle)

        if dist < 0.30 and abs(yaw_err) < 5.0:
            print("[PARK] Slot pose reached")
            break

        steer = max(-1.0, min(1.0, yaw_err / 20.0))

        if dist > 1.0:
            throttle = 0.16 if sp < 1.0 else 0.0
            brake = 0.0
        else:
            throttle = 0.08 if sp < 0.5 else 0.0
            brake = 0.0

        if abs(yaw_err) > 20 and sp > 0.8:
            throttle = 0.0
            brake = 0.2

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        ))
        time.sleep(0.05)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    print(f"[PARK] Final yaw: {vehicle.get_transform().rotation.yaw:.2f}")
    print("[PARK] Parked ✓")


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
slots_data = build_fixed_slots()

# def refresh_semantics_and_nav():
#     global semantic_grid, nav_grid, inflated_grid, slots_data

#     print("[GRID] Building semantic grid from CARLA world data…")
#     semantic_grid = build_semantic_grid(world, vehicle)

#     update_slot_occupancy(semantic_grid, slots_data)
#     n_occ = sum(1 for s in slots_data if s["occupied"])
#     n_free = sum(1 for s in slots_data if not s["occupied"])
#     print(f"[SLOT] total={len(slots_data)} free={n_free} occupied={n_occ}")

#     nav_grid = build_nav_grid_from_semantics(semantic_grid)
#     inflated_grid = inflate(nav_grid, INFLATION_RADIUS_M)
#     print(f"[GRID] Inflated occupied cells: {int(np.sum(inflated_grid != FREE))}")

def refresh_semantics_and_nav():
    global semantic_grid, nav_grid, inflated_grid, slots_data

    print("[GRID] Building semantic grid from occupancy_grid.py ...")
    semantic_grid = build_semantic_grid_truth(
        world,
        include_dynamic=True,
        ego_actor_id=vehicle.id
    )

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

            px, py = world_to_pixel(slot["cx"], slot["cy"])
            cv2.drawMarker(frame, (px, py), (0, 255, 0),
                        cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)

            apx, apy = world_to_pixel(slot["approach_x"], slot["approach_y"])
            cv2.circle(frame, (apx, apy), 8, (255, 255, 0), 2)

        
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
        demo_state["slot_id"] = slot["id"]
        demo_state["message"] = (
            f"Chosen slot #{slot['id']} at ({slot['cx']:.2f}, {slot['cy']:.2f}), "
            f"yaw={slot['yaw']:.1f}"
        )

        print(f"[SLOT] Chosen slot #{slot['id']} center=({slot['cx']:.2f}, {slot['cy']:.2f}) yaw={slot['yaw']:.1f}")
        draw_label(world, slot["cx"], slot["cy"], f"SLOT {slot['id']}", carla.Color(255, 0, 0))
        draw_label(world, slot["approach_x"], slot["approach_y"], "APPROACH", carla.Color(0, 255, 255))

        # Make small holes around approach and final center
        work_grid = inflated_grid.copy()
        for wx, wy in [(slot["approach_x"], slot["approach_y"]), (slot["cx"], slot["cy"])]:
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
        path = plan(work_grid, start_xy, (slot["approach_x"], slot["approach_y"]))
        draw_path(world, path, carla.Color(0, 200, 255))

        demo_state["message"] = "Driving to slot approach…"
        time.sleep(1.0)
        follow_path(vehicle, path, (slot["approach_x"], slot["approach_y"]))

        demo_state["message"] = "Pulling into slot…"
        pull_in_to_slot(vehicle, slot)

        tf = vehicle.get_transform()
        dist = math.hypot(tf.location.x - slot["cx"], tf.location.y - slot["cy"])
        yaw_err = abs(normalize_angle_deg(slot["yaw"] - tf.rotation.yaw))
        print(f"[PARK] Final pose=({tf.location.x:.2f}, {tf.location.y:.2f}) dist={dist:.2f} yaw_err={yaw_err:.2f}")

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
  #btn {
    margin-top:12px;
    padding:10px 24px;
    font-size:16px;
    background:#1a8;
    color:#fff;
    border:none;
    border-radius:6px;
    cursor:pointer;
  }
  #btn:hover { background:#1ca; }
  #btn:disabled { background:#666; cursor:not-allowed; }
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

<button id="btn" onclick="parkRandom()">🎲 Park in Random Empty Slot</button>
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