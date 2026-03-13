import carla
import random
import numpy as np
import cv2
import queue
import threading
from flask import Flask, Response

WIDTH, HEIGHT = 800, 600

TEST_SPAWN_INDEX = 5
USE_SPAWN_INDEX = True

# Use spawn 5 coord for initialization
SPAWN_X = -95.240860
SPAWN_Y = -88.037468
CAM_Z = 55.0

# set ego vehicle location
VEHICLE_X = SPAWN_X + 50
VEHICLE_Y = SPAWN_Y + 60
VEHICLE_Z = 0.5
VEHICLE_YAW = 0.0

# set camera up above parking lot
# X: +/- -> up/down
# Y: +/- -> right/left
CAM_X = SPAWN_X + 85
CAM_Y = SPAWN_Y + 60


app = Flask(__name__)
image_queue = queue.Queue(maxsize=1)
latest_jpeg = None

def sensor_callback(image):
    global latest_jpeg
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # BGR -> RGB-ish ordering for display
    ok, jpeg = cv2.imencode(".jpg", array)
    if ok:
        latest_jpeg = jpeg.tobytes()

@app.route("/")
def index():
    return """
    <html>
      <head><title>CARLA Remote View</title></head>
      <body>
        <h2>CARLA Remote Camera</h2>
        <img src="/video_feed" width="800" />
      </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    def gen():
        global latest_jpeg
        while True:
            if latest_jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + latest_jpeg + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def draw_lot_border(world, center_x, center_y, width, height, z=1.0):
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2

    p1 = carla.Location(x=x_min, y=y_min, z=z)
    p2 = carla.Location(x=x_max, y=y_min, z=z)
    p3 = carla.Location(x=x_max, y=y_max, z=z)
    p4 = carla.Location(x=x_min, y=y_max, z=z)

    color = carla.Color(0, 225, 0)

    life_time = 50.0

    world.debug.draw_line(p1, p2, thickness=0.15, color=color, life_time=life_time)
    world.debug.draw_line(p2, p3, thickness=0.15, color=color, life_time=life_time)
    world.debug.draw_line(p3, p4, thickness=0.15, color=color, life_time=life_time)
    world.debug.draw_line(p4, p1, thickness=0.15, color=color, life_time=life_time)

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # draw occupancy parking lot boarder
    LOT_CENTER_X = CAM_X
    LOT_CENTER_Y = CAM_Y
    LOT_WIDTH_M = 60.0
    LOT_HEIGHT_M = 60.0
    draw_lot_border(world, LOT_CENTER_X, LOT_CENTER_Y, LOT_WIDTH_M, LOT_HEIGHT_M)

    # draw parking lot center
    world.debug.draw_point(
        carla.Location(x=LOT_CENTER_X, y=LOT_CENTER_Y, z=1.0),
        size=0.3,
        color=carla.Color(0, 255, 0),
        life_time=0.0,
        persistent_lines=True
    )
    bp_lib = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()
    print("num spawn points:", len(spawn_points))

    if USE_SPAWN_INDEX:
        spawn_tf = carla.Transform(
            carla.Location(x=VEHICLE_X, y=VEHICLE_Y, z=VEHICLE_Z),
            carla.Rotation(yaw=VEHICLE_YAW)
        )
        print("using spawn index:", TEST_SPAWN_INDEX)
        print("spawn location:", spawn_tf.location)

        vehicle_bp = bp_lib.filter("vehicle.*")[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    else:
        vehicle_bp = bp_lib.filter("vehicle.*")[0]
        spawn_tf = carla.Transform(
            carla.Location(x=CAM_X, y=CAM_Y, z=0.5),
            carla.Rotation(yaw=0.0)
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn_tf)

    vehicle.set_autopilot(False)
    

    # # spawn_tf = spawn_points[0]   # or random.choice(spawn_points)
    # spawn_tf = spawn_points[TEST_SPAWN_INDEX]
    # print("using spawn index:", TEST_SPAWN_INDEX)
    # print("spawn location:", spawn_tf.location)
    
    # vehicle_bp = bp_lib.filter("vehicle.*")[0]
    # vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    # vehicle.set_autopilot(False)

    # spawn_points = world.get_map().get_spawn_points()
    # spawn_tf = random.choice(spawn_points)
    # vehicle_bp = bp_lib.filter("vehicle.*")[0]
    # vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    # vehicle.set_autopilot(True)

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(WIDTH))
    camera_bp.set_attribute("image_size_y", str(HEIGHT))
    camera_bp.set_attribute("fov", "100")

    # camera following ego vehicle
    # cam_tf = carla.Transform(
    #     carla.Location(x=-6.0, z=2.5),
    #     carla.Rotation(pitch=-10.0)
    # )

    # cam_tf = carla.Transform(
    #     carla.Location(
    #         x=spawn_tf.location.x,
    #         y=spawn_tf.location.y,
    #         z=CAMERA_HEIGHT
    #     ),
    #     carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    # )
    
    cam_tf = carla.Transform(
        carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(camera_bp, cam_tf)
    # camera = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)
    camera.listen(sensor_callback)

    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()

if __name__ == "__main__":
    main()
