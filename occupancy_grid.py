import carla
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
LOT_CENTER_X = -95.24+85
LOT_CENTER_Y = -88.04+60
LOT_WIDTH_M = 60.0
LOT_HEIGHT_M = 60.0
RESOLUTION = 0.5

SAVE_PATH = "/scratch/tsethi2/occupancy_grid/town05_semantic.png"

# semantic values
OUTSIDE = 0
DRIVABLE = 1
STATIC_CAR = 2
DYNAMIC_CAR = 3
ROAD_LINE = 4

GRID_W = int(LOT_WIDTH_M / RESOLUTION)
GRID_H = int(LOT_HEIGHT_M / RESOLUTION)


# =========================
# COORDINATE HELPERS
# =========================
def world_to_grid(x, y):
    x_min = LOT_CENTER_X - LOT_WIDTH_M / 2
    y_min = LOT_CENTER_Y - LOT_HEIGHT_M / 2

    col = int((x - x_min) / RESOLUTION)
    row = int((y - y_min) / RESOLUTION)
    return row, col


def in_grid(row, col):
    return 0 <= row < GRID_H and 0 <= col < GRID_W


# =========================
# DEBUG DRAWING
# =========================
def draw_lot_border(world, cx, cy, w, h, z=1.0, life_time=1.0):
    x_min = cx - w / 2
    x_max = cx + w / 2
    y_min = cy - h / 2
    y_max = cy + h / 2

    p1 = carla.Location(x=x_min, y=y_min, z=z)
    p2 = carla.Location(x=x_max, y=y_min, z=z)
    p3 = carla.Location(x=x_max, y=y_max, z=z)
    p4 = carla.Location(x=x_min, y=y_max, z=z)

    color = carla.Color(0, 255, 0)

    world.debug.draw_line(p1, p2, thickness=0.15, color=color, life_time=life_time)
    world.debug.draw_line(p2, p3, thickness=0.15, color=color, life_time=life_time)
    world.debug.draw_line(p3, p4, thickness=0.15, color=color, life_time=life_time)
    world.debug.draw_line(p4, p1, thickness=0.15, color=color, life_time=life_time)

    world.debug.draw_point(
        carla.Location(x=cx, y=cy, z=z),
        size=0.25,
        color=carla.Color(255, 0, 0),
        life_time=life_time,
    )


# =========================
# RASTERIZATION HELPERS
# =========================
def rasterize_bbox(grid, bbox, value):
    """
    Rasterize a CARLA bounding box into the grid using its world vertices.
    """
    verts = bbox.get_world_vertices(carla.Transform())

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    r0, c0 = world_to_grid(min_x, min_y)
    r1, c1 = world_to_grid(max_x, max_y)

    rmin, rmax = sorted([r0, r1])
    cmin, cmax = sorted([c0, c1])

    rmin = max(rmin, 0)
    rmax = min(rmax, GRID_H - 1)
    cmin = max(cmin, 0)
    cmax = min(cmax, GRID_W - 1)

    if rmin <= rmax and cmin <= cmax:
        grid[rmin:rmax + 1, cmin:cmax + 1] = value


def rasterize_actor_bbox(grid, actor, value):
    """
    Rasterize a live actor bounding box into the grid.
    """
    bb = actor.bounding_box
    tf = actor.get_transform()
    verts = bb.get_world_vertices(tf)

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    r0, c0 = world_to_grid(min_x, min_y)
    r1, c1 = world_to_grid(max_x, max_y)

    rmin, rmax = sorted([r0, r1])
    cmin, cmax = sorted([c0, c1])

    rmin = max(rmin, 0)
    rmax = min(rmax, GRID_H - 1)
    cmin = max(cmin, 0)
    cmax = min(cmax, GRID_W - 1)

    if rmin <= rmax and cmin <= cmax:
        grid[rmin:rmax + 1, cmin:cmax + 1] = value


# =========================
# LAYERS
# =========================
def mark_drivable_cells(world, grid):
    """
    Mark cells as drivable if they belong to Driving or Parking lane types.
    """
    carla_map = world.get_map()

    x_min = LOT_CENTER_X - LOT_WIDTH_M / 2
    y_min = LOT_CENTER_Y - LOT_HEIGHT_M / 2

    lane_mask = carla.LaneType.Driving | carla.LaneType.Parking

    for row in range(GRID_H):
        for col in range(GRID_W):
            wx = x_min + (col + 0.5) * RESOLUTION
            wy = y_min + (row + 0.5) * RESOLUTION

            loc = carla.Location(x=wx, y=wy, z=0.5)
            wp = carla_map.get_waypoint(
                loc,
                project_to_road=False,
                lane_type=lane_mask
            )

            if wp is not None:
                grid[row, col] = DRIVABLE


def bbox_center_in_lot(bb, margin=2.0):
    x = bb.location.x
    y = bb.location.y

    x_min = LOT_CENTER_X - LOT_WIDTH_M / 2 - margin
    x_max = LOT_CENTER_X + LOT_WIDTH_M / 2 + margin
    y_min = LOT_CENTER_Y - LOT_HEIGHT_M / 2 - margin
    y_max = LOT_CENTER_Y + LOT_HEIGHT_M / 2 + margin

    return x_min <= x <= x_max and y_min <= y <= y_max


def mark_static_vehicles(world, grid):
    """
    Mark static map vehicles using multiple semantic labels.
    """
    labels = [
        carla.CityObjectLabel.Car,
        carla.CityObjectLabel.Truck,
        carla.CityObjectLabel.Bus,
        carla.CityObjectLabel.Motorcycle,
        carla.CityObjectLabel.Bicycle,
    ]

    total = 0

    for label in labels:
        try:
            bbs = world.get_level_bbs(label)
        except Exception as e:
            print(f"Could not query {label}: {e}")
            bbs = []

        # keep only boxes near the selected lot
        bbs = [bb for bb in bbs if bbox_center_in_lot(bb)]

        print(f"{label}: {len(bbs)} static boxes in/near lot")
        total += len(bbs)

        for bb in bbs:
            rasterize_bbox(grid, bb, STATIC_CAR)

    print("Total static vehicle boxes used:", total)


def mark_dynamic_cars(world, grid):
    """
    Mark live vehicle actors.
    """
    vehicles = world.get_actors().filter("vehicle.*")
    print("Live vehicle actors:", len(vehicles))

    for v in vehicles:
        print(
            v.id,
            v.type_id,
            v.get_location(),
            "extent:", v.bounding_box.extent
        )
        rasterize_actor_bbox(grid, v, DYNAMIC_CAR)


def mark_road_lines(world, grid):
    """
    Optional: mark road/parking lines using semantic level bounding boxes.
    """
    try:
        road_line_bbs = world.get_level_bbs(carla.CityObjectLabel.RoadLines)
    except Exception as e:
        print("Could not query road line bounding boxes:", e)
        road_line_bbs = []

    print("Road line bounding boxes:", len(road_line_bbs))

    for bb in road_line_bbs:
        # only paint lines where grid is already drivable
        temp = np.zeros_like(grid)
        rasterize_bbox(temp, bb, ROAD_LINE)
        mask = (temp == ROAD_LINE) & (grid == DRIVABLE)
        grid[mask] = ROAD_LINE

# =========================
# NEW: ACCURACY ADDITIONS
# =========================
def mark_road_lines_high_accuracy(world, grid):
    """
    Ensures individual lines are painted clearly to match visual plots.
    """
    try:
        road_line_bbs = [bb for bb in world.get_level_bbs(carla.CityObjectLabel.RoadLines) if bbox_center_in_lot(bb)]
        for bb in road_line_bbs:
            # Bypass the mask and paint lines directly for maximum visibility
            rasterize_bbox(grid, bb, ROAD_LINE)
            
            # If it's a parking-sized line, also mark the slot orientation
            ext = bb.extent
            if 4.0 < max(ext.x, ext.y) * 2 < 6.0:
                tf = carla.Transform(bb.location, bb.rotation)
                # Mark a small orientation "tail" to define the slot space
                slot_loc = bb.location + carla.Location(x=tf.get_right_vector().x * 1.5, y=tf.get_right_vector().y * 1.5)
                r, c = world_to_grid(slot_loc.x, slot_loc.y)
                if in_grid(r, c):
                    grid[r, c] = ROAD_LINE
    except:
        pass


# =========================
# VISUALIZATION
# =========================
def save_grid_image(grid):
    """
    Save two versions:
    - raw planning grid
    - display-aligned grid for easier comparison with browser view
    """
    # raw
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", cmap="viridis")
    plt.title("Semantic Occupancy Grid (raw)")
    plt.colorbar()
    plt.savefig(SAVE_PATH.replace(".png", "_raw.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # display-aligned version
    grid_vis = np.rot90(grid, k=1)
    grid_vis = np.flipud(grid_vis)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_vis, origin="lower", cmap="viridis")
    plt.title("Semantic Occupancy Grid (display-aligned)")
    plt.colorbar()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(" ", SAVE_PATH.replace(".png", "_raw.png"))
    print(" ", SAVE_PATH)


# =========================
# MAIN
# =========================
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    print("Map:", world.get_map().name)

    draw_lot_border(world, LOT_CENTER_X, LOT_CENTER_Y, LOT_WIDTH_M, LOT_HEIGHT_M)

    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)

    # 1) drivable / parking area
    mark_drivable_cells(world, grid)

    # 2) static parked cars / static car meshes
    mark_static_vehicles(world, grid)

    # 3) live vehicle actors (ego or spawned cars)
    mark_dynamic_cars(world, grid)

    # 4) original road line overlay
    mark_road_lines(world, grid)
    
    # 5) HIGH ACCURACY ADDITION
    # This step ensures the individual plots show up clearly as in your reference image
    mark_road_lines_high_accuracy(world, grid)

    save_grid_image(grid)


if __name__ == "__main__":
    main()