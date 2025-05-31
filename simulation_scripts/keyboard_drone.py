import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import threading
import time
from pynput import keyboard

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

velocity = 2  # m/s
duration = 0.5  # movement duration
running = True

def move_drone(direction):
    if direction == 'forward':
        client.moveByVelocityAsync(velocity, 0, 0, duration).join()
    elif direction == 'backward':
        client.moveByVelocityAsync(-velocity, 0, 0, duration).join()
    elif direction == 'left':
        client.moveByVelocityAsync(0, -velocity, 0, duration).join()
    elif direction == 'right':
        client.moveByVelocityAsync(0, velocity, 0, duration).join()
    elif direction == 'up':
        client.moveByVelocityAsync(0, 0, -velocity, duration).join()
    elif direction == 'down':
        client.moveByVelocityAsync(0, 0, velocity, duration).join()

def on_press(key):
    global running
    try:
        if key == keyboard.Key.up:
            move_drone('forward')
        elif key == keyboard.Key.down:
            move_drone('backward')
        elif key == keyboard.Key.left:
            move_drone('left')
        elif key == keyboard.Key.right:
            move_drone('right')
        elif key.char == 'u':
            move_drone('up')
        elif key.char == 'd':
            move_drone('down')
        elif key.char == 'q':
            running = False
            return False  # stop listener
    except AttributeError:
        pass

print("Control the drone using arrow keys. Use 'U' to go up, 'D' to go down. Press 'Q' to quit.")

listener = keyboard.Listener(on_press=on_press)
listener.start()

while running:
    time.sleep(0.1)

print("Landing and cleaning up...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
