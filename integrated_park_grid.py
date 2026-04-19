import carla
import math
import heapq
import queue
import threading
import time
import os

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

# =========================
# CONFIG
# =========================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5000

MAP_NAME = "Town05"

GRID_RES = 0.5
INFLATION_RADIUS_M = 1.5

FREE = 0
OCCUPIED = 1
INFLATED = 2

# =========================
# LOT DEFINITION  ← must match occupancy_grid.py exactly
# =========================
LOT_CENTER_X = -95.24 + 85   # -10.24
LOT_CENTER_Y = -88.04 + 60   # -28.04
LOT_WIDTH_M  = 60.0
LOT_HEIGHT_M = 60.0

# Nav-grid is the same size as the lot grid in occupancy_grid.py.
# grid[row, col] where row = Y-axis index, col = X-axis index.
# This matches how occupancy_grid.py lays out its grid.
GRID_W = int(LOT_WIDTH_M  / GRID_RES)   # 120  (X / col axis)
GRID_H = int(LOT_HEIGHT_M / GRID_RES)   # 120  (Y / row axis)

# Bottom-left corner of the grid in world space
ORIGIN_X = LOT_CENTER_X - LOT_WIDTH_M  / 2   # -40.24
ORIGIN_Y = LOT_CENTER_Y - LOT_HEIGHT_M / 2   # -58.04

GOAL_REACHED_DIST = 0.4

# Static aerial camera — MUST match these exactly for pixel→world to work
CAMERA_X     = -45.0
CAMERA_Y     = -25.0
CAMERA_Z     =  55.0
CAM_IMG_W    = 1000
CAM_IMG_H    =  700
CAM_HFOV_DEG =  90.0

# Derived spans (orthographic approximation)
_half_hfov = math.radians(CAM_HFOV_DEG / 2)
CAM_SPAN_Y = 2 * CAMERA_Z * math.tan(_half_hfov)
CAM_SPAN_X = 2 * CAMERA_Z * math.tan(
    math.atan(math.tan(_half_hfov) * CAM_IMG_H / CAM_IMG_W))

# Spawn: inside the parking lot aisle
SPAWN_X   = -35.21
SPAWN_Y   = -48.03
SPAWN_YAW =  90.0

# =========================
# SHARED STATE
# =========================
slot_target   = {"x": None, "y": None}
demo_state    = {"phase": "waiting"}
_demo_trigger = threading.Event()


def pixel_to_world(px, py):
    """Convert a click on the 1000×700 stream image to CARLA world XY."""
    wx = CAMERA_X + (0.5 - py / CAM_IMG_H) * CAM_SPAN_X
    wy = CAMERA_Y + (px / CAM_IMG_W - 0.5) * CAM_SPAN_Y
    return round(wx, 2), round(wy, 2)


# =========================
# GRID / PATH HELPERS
# =========================
# IMPORTANT: park14 convention is grid[gy, gx] where gx=col=X, gy=row=Y.
# This matches occupancy_grid.py which uses grid[row, col] with row=Y, col=X.
# world_to_grid returns (gx, gy) i.e. (col, row).

def world_to_grid(x, y, ox, oy):
    """World XY → (gx, gy) = (col, row). Matches park14 original convention."""
    return int((x - ox) / GRID_RES), int((y - oy) / GRID_RES)

def grid_to_world(gx, gy, ox, oy):
    return ox + (gx + 0.5) * GRID_RES, oy + (gy + 0.5) * GRID_RES

def in_bounds(gx, gy):
    return 0 <= gx < GRID_W and 0 <= gy < GRID_H

def heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def normalize_angle_deg(a):
    while a >  180: a -= 360
    while a < -180: a += 360
    return a

def get_speed(v):
    vel = v.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


# =========================
# OCCUPANCY GRID BUILDER
# Mirrors occupancy_grid.py exactly — queries CARLA directly.
# No image parsing needed; this is coordinate-exact and always correct.
# =========================

def _mark_world_point(grid, wx, wy, val):
    """Write val at the cell that contains world point (wx, wy)."""
    gx, gy = world_to_grid(wx, wy, ORIGIN_X, ORIGIN_Y)
    if in_bounds(gx, gy):
        grid[gy, gx] = val


def _bbox_center_in_lot(bb, margin=2.0):
    x, y = bb.location.x, bb.location.y
    return (ORIGIN_X - margin <= x <= ORIGIN_X + LOT_WIDTH_M  + margin and
            ORIGIN_Y - margin <= y <= ORIGIN_Y + LOT_HEIGHT_M + margin)


def _rasterize_bbox_world(grid, bb_location, bb_extent, val):
    """
    Axis-aligned bounding-box rasteriser.
    Works with a bare (location, extent) pair — no CARLA Transform needed.
    """
    min_x = bb_location.x - bb_extent.x
    max_x = bb_location.x + bb_extent.x
    min_y = bb_location.y - bb_extent.y
    max_y = bb_location.y + bb_extent.y

    gx0, gy0 = world_to_grid(min_x, min_y, ORIGIN_X, ORIGIN_Y)
    gx1, gy1 = world_to_grid(max_x, max_y, ORIGIN_X, ORIGIN_Y)

    gx_min, gx_max = sorted([gx0, gx1])
    gy_min, gy_max = sorted([gy0, gy1])

    gx_min = max(gx_min, 0);  gx_max = min(gx_max, GRID_W - 1)
    gy_min = max(gy_min, 0);  gy_max = min(gy_max, GRID_H - 1)

    if gx_min <= gx_max and gy_min <= gy_max:
        grid[gy_min:gy_max+1, gx_min:gx_max+1] = val


def build_static_grid(world):
    """
    Build the base occupancy grid from CARLA world data.
    Replicates what occupancy_grid.py does:
      1. Mark road-line bounding boxes as OCCUPIED.
      2. Mark static vehicle bounding boxes as OCCUPIED.
    Drivable / outside cells are left FREE — the car can drive anywhere
    that isn't a parked car or a floor line.
    """
    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)

    # ── 1. Road lines ────────────────────────────────────────────────────
    try:
        road_line_bbs = world.get_level_bbs(carla.CityObjectLabel.RoadLines)
        road_line_bbs = [bb for bb in road_line_bbs if _bbox_center_in_lot(bb)]
        print(f"[GRID] Road-line bboxes in lot: {len(road_line_bbs)}")
        for bb in road_line_bbs:
            _rasterize_bbox_world(grid, bb.location, bb.extent, OCCUPIED)
    except Exception as e:
        print(f"[GRID] Could not query road lines: {e}")

    # ── 2. Static vehicles ───────────────────────────────────────────────
    labels = [
        carla.CityObjectLabel.RoadLines,
        carla.CityObjectLabel.Sidewalks,
        carla.CityObjectLabel.Walls,
        carla.CityObjectLabel.Fences,
        carla.CityObjectLabel.Car,
        carla.CityObjectLabel.Truck,
        carla.CityObjectLabel.Bus,
        carla.CityObjectLabel.Motorcycle,
        carla.CityObjectLabel.Bicycle,
    ]
    total_static = 0
    for label in labels:
        try:
            bbs = world.get_level_bbs(label)
            bbs = [bb for bb in bbs if _bbox_center_in_lot(bb)]
            total_static += len(bbs)
            for bb in bbs:
                _rasterize_bbox_world(grid, bb.location, bb.extent, OCCUPIED)
        except Exception as e:
            print(f"[GRID] Could not query {label}: {e}")
    print(f"[GRID] Static vehicle bboxes in lot: {total_static}")

    n_obs = int(np.sum(grid == OCCUPIED))
    print(f"[GRID] Total occupied cells after static build: {n_obs}")
    border_px = int(1.0 / GRID_RES)
    grid[0:border_px, :] = OCCUPIED           # Bottom edge
    grid[-border_px:, :] = OCCUPIED           # Top edge
    grid[:, 0:border_px] = OCCUPIED           # Left edge
    grid[:, -border_px:] = OCCUPIED           # Right edge
    return grid


# =========================
# A* AND PLANNING
# =========================
def astar(grid, start, goal):
    moves = [(-1,0,1),( 1,0,1),(0,-1,1),(0,1,1),
             (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),
             ( 1,-1,math.sqrt(2)),( 1,1,math.sqrt(2))]
    heap = [(heuristic(start,goal), 0.0, start)]
    came_from, g = {}, {start: 0.0}
    visited = set()
    while heap:
        _, cg, cur = heapq.heappop(heap)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal:
            path = [cur]
            while cur in came_from: cur=came_from[cur]; path.append(cur)
            return path[::-1]
        cx, cy = cur
        for dx, dy, sc in moves:
            nb = (cx+dx, cy+dy)
            if not in_bounds(*nb): continue
            if grid[nb[1],nb[0]] in (OCCUPIED,INFLATED): continue
            tg = cg + sc
            if tg < g.get(nb, 1e18):
                g[nb] = tg; came_from[nb] = cur
                heapq.heappush(heap, (tg+heuristic(nb,goal), tg, nb))
    return None

def nearest_free(grid, gx, gy, r=20):
    if in_bounds(gx,gy) and grid[gy,gx]==FREE: return (gx,gy)
    for rad in range(1,r+1):
        for dy in range(-rad,rad+1):
            for dx in range(-rad,rad+1):
                nx,ny=gx+dx,gy+dy
                if in_bounds(nx,ny) and grid[ny,nx]==FREE: return (nx,ny)
    return None

def draw_rotated_box(grid, cx, cy, yaw_rad, hl, hw, ox, oy, val):
    c,s = math.cos(yaw_rad), math.sin(yaw_rad)
    wc = [(cx+dx*c-dy*s, cy+dx*s+dy*c) for dx,dy in
          [(+hl,+hw),(+hl,-hw),(-hl,-hw),(-hl,+hw)]]
    xs=[p[0] for p in wc]; ys=[p[1] for p in wc]
    mgx,mgy=world_to_grid(min(xs),min(ys),ox,oy)
    Mgx,Mgy=world_to_grid(max(xs),max(ys),ox,oy)
    for gx in range(max(0,mgx),min(GRID_W,Mgx+1)):
        for gy in range(max(0,mgy),min(GRID_H,Mgy+1)):
            wx,wy=grid_to_world(gx,gy,ox,oy)
            lx=(wx-cx)*c+(wy-cy)*s; ly=-(wx-cx)*s+(wy-cy)*c
            if abs(lx)<=hl and abs(ly)<=hw: grid[gy,gx]=val

def inflate(grid, radius_m):
    out = grid.copy()
    r = int(math.ceil(radius_m/GRID_RES))
    for gy,gx in np.argwhere(grid==OCCUPIED):
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                nx,ny=gx+dx,gy+dy
                if in_bounds(nx,ny) and dx*dx+dy*dy<=r*r and out[ny,nx]==FREE:
                    out[ny,nx]=INFLATED
    return out

def sparsify(path, step=4):
    if len(path)<=2: return path
    out=[path[0]]
    for i in range(step,len(path),step): out.append(path[i])
    if out[-1]!=path[-1]: out.append(path[-1])
    return out

def plan(grid, ox, oy, start_xy, goal_xy):
    s=world_to_grid(*start_xy,ox,oy); g=world_to_grid(*goal_xy,ox,oy)
    if not in_bounds(*s): raise RuntimeError(f"Start {start_xy} outside grid (gx={s[0]},gy={s[1]})")
    if not in_bounds(*g): raise RuntimeError(f"Goal {goal_xy} outside grid (gx={g[0]},gy={g[1]})")
    sf=nearest_free(grid,*s,20); gf=nearest_free(grid,*g,20)
    if not sf: raise RuntimeError("No free start cell near spawn")
    if not gf: raise RuntimeError("No free goal cell near target")
    path=astar(grid,sf,gf)
    if not path: raise RuntimeError("A* found no path")
    wp=[grid_to_world(gx,gy,ox,oy) for gx,gy in path]
    return sparsify(wp,4)

def draw_path(world_obj, path, color, z=0.5):
    for i in range(len(path)-1):
        x1,y1=path[i]; x2,y2=path[i+1]
        world_obj.debug.draw_line(
            carla.Location(x=x1,y=y1,z=z), carla.Location(x=x2,y=y2,z=z),
            thickness=0.12, color=color, life_time=120.0)

def draw_label(world_obj, x, y, text, color):
    world_obj.debug.draw_string(
        carla.Location(x=x,y=y,z=1.5), text,
        draw_shadow=False, color=color, life_time=120.0)


# =========================
# CONTROLLERS
# =========================
def follow_path(vehicle, path, goal_xy):
    lookahead = 2
    while True:
        t = vehicle.get_transform()
        cx, cy, yaw = t.location.x, t.location.y, t.rotation.yaw

        ci = min(range(len(path)), key=lambda i: math.hypot(path[i][0]-cx, path[i][1]-cy))
        ti = min(ci + lookahead, len(path)-1)
        wx, wy = path[ti]

        dist_to_goal = math.hypot(goal_xy[0]-cx, goal_xy[1]-cy)

        if dist_to_goal < 0.8:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            break

        ye = normalize_angle_deg(math.degrees(math.atan2(wy-cy, wx-cx)) - yaw)
        st = max(-1.0, min(1.0, ye / 40.0))
        sp = get_speed(vehicle)

        if dist_to_goal < 5.0:
            th = 0.12 if sp < 1.0 else 0.0
            if sp > 1.5:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3))
                time.sleep(0.05)
                continue
        else:
            th = 0.3 if sp < 3.0 else 0.0
            if abs(ye) > 30: th = 0.15

        vehicle.apply_control(carla.VehicleControl(throttle=th, steer=st, brake=0.0))
        time.sleep(0.05)

def pull_in(vehicle, slot_x, slot_y, extra=2.5):
    t = vehicle.get_transform()
    current_yaw = t.rotation.yaw

    target_yaw = 0.0 if abs(normalize_angle_deg(current_yaw - 90)) < 90 else 180.0

    rad = math.radians(target_yaw)
    deep_x = slot_x + extra * math.cos(rad)
    deep_y = slot_y + extra * math.sin(rad)

    deadline = time.time() + 10.0
    while time.time() < deadline:
        t = vehicle.get_transform()
        cx, cy, yaw = t.location.x, t.location.y, t.rotation.yaw

        dist_to_actual_slot = math.hypot(slot_x - cx, slot_y - cy)
        yaw_error = abs(normalize_angle_deg(target_yaw - yaw))

        if dist_to_actual_slot < 0.5 and yaw_error < 3.0:
            print("Target orientation reached!")
            break
        if dist_to_actual_slot < 0.15:
            print("Distance limit reached (Safety Stop)")
            break

        st = max(-1.0, min(1.0, normalize_angle_deg(target_yaw - yaw) / 15.0))
        sp = get_speed(vehicle)
        th = 0.13 if sp < 0.7 else 0.0

        vehicle.apply_control(carla.VehicleControl(throttle=th, steer=st, brake=0.0))
        time.sleep(0.05)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    print(f"Parked at Angle: {vehicle.get_transform().rotation.yaw:.2f}")
    print("Parked ✓")


# =========================
# CONNECT TO CARLA
# =========================
client=carla.Client(CARLA_HOST,CARLA_PORT); client.set_timeout(20.0)
print(f"Loading {MAP_NAME}…")
world=client.load_world(MAP_NAME); time.sleep(2.0)

bp_lib=world.get_blueprint_library()
vehicle_bp=bp_lib.find("vehicle.tesla.model3")

spawn_tf=carla.Transform(
    carla.Location(x=SPAWN_X,y=SPAWN_Y,z=0.5),
    carla.Rotation(yaw=SPAWN_YAW))
vehicle=world.try_spawn_actor(vehicle_bp,spawn_tf)

if vehicle is None:
    for dx,dy in [(2,0),(-2,0),(0,2),(0,-2),(2,2),(-2,-2)]:
        spawn_tf.location.x=SPAWN_X+dx; spawn_tf.location.y=SPAWN_Y+dy
        vehicle=world.try_spawn_actor(vehicle_bp,spawn_tf)
        if vehicle: break
if vehicle is None:
    raise RuntimeError("Could not spawn. Adjust SPAWN_X/Y/YAW.")

vehicle.set_autopilot(False)
vehicle.set_simulate_physics(True)
vehicle.apply_control(carla.VehicleControl(throttle=0.0,steer=0.0,brake=0.0))
time.sleep(1.0)

t=vehicle.get_transform()
ego_x,ego_y=t.location.x,t.location.y
print(f"Spawned at ({ego_x:.2f}, {ego_y:.2f})")
print(f"Grid origin: ({ORIGIN_X:.2f}, {ORIGIN_Y:.2f}), size: {GRID_W}x{GRID_H} cells")

# Sanity-check spawn is inside grid
_sgx, _sgy = world_to_grid(ego_x, ego_y, ORIGIN_X, ORIGIN_Y)
print(f"Spawn grid cell: gx={_sgx}, gy={_sgy}  (must be 0–{GRID_W-1} and 0–{GRID_H-1})")

draw_label(world,ego_x,ego_y,"START",carla.Color(0,255,0))

# =========================
# BUILD OCCUPANCY GRID  (direct CARLA query — no image parsing)
# =========================
# The grid origin is fixed to the lot corner defined by LOT_CENTER + extents.
# This is identical to how occupancy_grid.py defines its coordinate space.
origin_x = ORIGIN_X
origin_y = ORIGIN_Y

print("Building static occupancy grid from CARLA world data…")
grid = build_static_grid(world)

# Stamp any live vehicle actors (excluding ego) on top
for actor in world.get_actors().filter("vehicle.*"):
    if actor.id==vehicle.id: continue
    at=actor.get_transform(); bb=actor.bounding_box
    draw_rotated_box(grid,at.location.x,at.location.y,
                     math.radians(at.rotation.yaw),
                     bb.extent.x+0.3,
                     bb.extent.y+0.3,
                     origin_x,origin_y,OCCUPIED)

inflated_grid=inflate(grid,INFLATION_RADIUS_M)
print(f"[GRID] Inflated occupied cells: {int(np.sum(inflated_grid != FREE))}")

# =========================
# CAMERA
# =========================
cam_bp=bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x",str(CAM_IMG_W))
cam_bp.set_attribute("image_size_y",str(CAM_IMG_H))
cam_bp.set_attribute("fov",str(CAM_HFOV_DEG))
camera=world.spawn_actor(cam_bp,carla.Transform(
    carla.Location(x=CAMERA_X,y=CAMERA_Y,z=CAMERA_Z),
    carla.Rotation(pitch=-90.0,yaw=0.0,roll=0.0)))

image_queue=queue.Queue(maxsize=1)

def process_image(image):
    arr=np.frombuffer(image.raw_data,dtype=np.uint8)
    frame=arr.reshape((image.height,image.width,4))[:,:,:3].copy()

    sx,sy=slot_target["x"],slot_target["y"]
    if sx is not None:
        px=int((sy-CAMERA_Y)/CAM_SPAN_Y*CAM_IMG_W + CAM_IMG_W/2)
        py=int((CAMERA_X-sx)/CAM_SPAN_X*CAM_IMG_H + CAM_IMG_H/2)
        cv2.drawMarker(frame,(px,py),(0,0,255),
                       cv2.MARKER_CROSS,30,3,cv2.LINE_AA)

    if not image_queue.empty():
        try: image_queue.get_nowait()
        except queue.Empty: pass
    image_queue.put(frame)

camera.listen(process_image)

# =========================
# DEMO THREAD
# =========================
def run_demo():
    print("Waiting for you to click a parking slot in the browser…")
    _demo_trigger.wait()

    sx,sy=slot_target["x"],slot_target["y"]
    print(f"Target slot: ({sx}, {sy})")
    demo_state["phase"]="running"

    draw_label(world,sx,sy,"SLOT",carla.Color(255,0,0))

    # Punch a hole at the target so inflation doesn't block the final cell
    gx,gy=world_to_grid(sx,sy,origin_x,origin_y)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if in_bounds(gx+dx, gy+dy):
                inflated_grid[gy+dy, gx+dx] = FREE

    dx = sx - ego_x
    dy = sy - ego_y
    dist = math.hypot(dx, dy)

    stop_early_dist = 2.5
    target_dist = dist - stop_early_dist
    ratio = target_dist / dist
    sx_adj = ego_x + dx * ratio
    sy_adj = ego_y + dy * ratio

    path = plan(inflated_grid, origin_x, origin_y, (ego_x, ego_y), (sx_adj, sy_adj))
    draw_path(world, path, carla.Color(0,200,255))

    time.sleep(1.0)
    follow_path(vehicle, path, (sx_adj, sy_adj))
    pull_in(vehicle, sx_adj, sy_adj)

    tf=vehicle.get_transform()
    dist=math.hypot(tf.location.x-sx,tf.location.y-sy)
    print(f"Final: ({tf.location.x:.2f},{tf.location.y:.2f})  dist={dist:.2f} m")
    demo_state["phase"]="done"

threading.Thread(target=run_demo,daemon=True).start()

# =========================
# FLASK
# =========================
app=Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>CARLA Parking</title>
<style>
  body { background:#111; color:#eee; font-family:sans-serif;
         display:flex; flex-direction:column; align-items:center; }
  h2   { margin:12px 0 4px; }
  #info{ font-size:14px; margin-bottom:8px; color:#8ef; }
  #wrap{ position:relative; cursor:crosshair; }
  #wrap img { display:block; }
  #crosshair { position:absolute; pointer-events:none;
               border:2px solid red; border-radius:50%;
               width:20px; height:20px; display:none;
               transform:translate(-50%,-50%); }
  #btn  { margin-top:10px; padding:8px 24px; font-size:15px;
          background:#1a8; color:#fff; border:none; border-radius:6px;
          cursor:pointer; display:none; }
  #btn:hover { background:#1ca; }
  #status { margin-top:6px; font-size:13px; color:#fa8; }
</style>
</head>
<body>
<h2>CARLA Parking Demo</h2>
<div id="info">👆 Click on an empty parking space to set the target slot</div>
<div id="wrap">
  <img src="/video_feed" width="1000" id="feed">
  <div id="crosshair"></div>
</div>
<button id="btn" onclick="confirmSlot()">✅ Park Here</button>
<div id="status" id="status"></div>

<script>
let pending = null;

document.getElementById('feed').addEventListener('click', function(e) {
  const rect = this.getBoundingClientRect();
  const scaleX = 1000 / rect.width;
  const scaleY = 700  / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top)  * scaleY;

  fetch('/preview_slot?px='+px+'&py='+py)
    .then(r=>r.json()).then(d=>{
      pending = d;
      document.getElementById('info').textContent =
        '📍 Slot preview: world (' + d.wx + ', ' + d.wy + ')  — click "Park Here" to confirm';
      const ch = document.getElementById('crosshair');
      ch.style.left = (px / scaleX) + 'px';
      ch.style.top  = (py / scaleY) + 'px';
      ch.style.display = 'block';
      document.getElementById('btn').style.display = 'inline-block';
    });
});

function confirmSlot() {
  if (!pending) return;
  fetch('/set_slot', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(pending)
  }).then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = d.message;
    document.getElementById('btn').style.display = 'none';
    document.getElementById('info').textContent =
      '🚗 Driving to (' + pending.wx + ', ' + pending.wy + ')…';
    pollStatus();
  });
}

function pollStatus() {
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = 'Phase: ' + d.phase;
    if (d.phase !== 'done') setTimeout(pollStatus, 1000);
    else document.getElementById('info').textContent = '✅ Parked!';
  });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

@app.route("/preview_slot")
def preview_slot():
    px=float(request.args["px"]); py=float(request.args["py"])
    wx,wy=pixel_to_world(px,py)
    return jsonify({"px":px,"py":py,"wx":wx,"wy":wy})

@app.route("/set_slot", methods=["POST"])
def set_slot():
    data=request.get_json()
    wx,wy=float(data["wx"]),float(data["wy"])
    slot_target["x"]=wx; slot_target["y"]=wy
    _demo_trigger.set()
    return jsonify({"message":f"Slot set to ({wx}, {wy}). Driving now…"})

@app.route("/status")
def status():
    return jsonify(demo_state)

def gen_frames():
    while True:
        frame=image_queue.get()
        ok,buf=cv2.imencode(".jpg",frame)
        if not ok: continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

try:
    print(f"Open  http://{STREAM_HOST}:{STREAM_PORT}  then click an empty bay")
    app.run(host=STREAM_HOST,port=STREAM_PORT,threaded=True)
finally:
    for actor in [camera,vehicle]:
        try: actor.stop()
        except: pass
        try: actor.destroy()
        except: pass
