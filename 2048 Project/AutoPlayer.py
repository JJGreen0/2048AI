from PIL import ImageGrab
import pytesseract
import numpy as np
import time
import math
import keyboard
import pyautogui
from stable_baselines3 import PPO

# Path to the trained model
MODEL_PATH = '2048 Project/SavedModels/PPO2048.zip'

# Bounding box for the game screen (left, top, right, bottom)
GAME_BBOX = (700, 314, 1204, 816)

# Load the trained PPO model
model = PPO.load(MODEL_PATH)


def capture_grid():
    """Capture the game area and return the grid as integers."""
    image = ImageGrab.grab(bbox=GAME_BBOX)
    image = image.convert('L')  # grayscale
    tile_width = image.width // 4
    tile_height = image.height // 4
    grid = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            left = j * tile_width
            top = i * tile_height
            tile = image.crop((left, top, left + tile_width, top + tile_height))
            text = pytesseract.image_to_string(
                tile,
                config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
            )
            grid[i][j] = int(text.strip()) if text.strip().isdigit() else 0
    return grid


def grid_to_obs(grid):
    """Convert a 4x4 grid of integers into the one-hot encoded observation."""
    obs = np.zeros((4, 4, 12), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            val = grid[i][j]
            if val == 0:
                obs[i][j][0] = 1
            else:
                index = int(math.log2(val))
                if index < 12:
                    obs[i][j][index] = 1
    return obs


def predict_action():
    grid = capture_grid()
    obs = grid_to_obs(grid)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def press_action(action):
    keys = ['up', 'right', 'down', 'left']
    if 0 <= action < len(keys):
        pyautogui.press(keys[action])


print("Press 'q' to capture screen and play. Press 'esc' to exit.")
while True:
    if keyboard.is_pressed('q'):
        act = predict_action()
        press_action(act)
        time.sleep(0.2)  # small delay to avoid multiple presses
    if keyboard.is_pressed('esc'):
        break
    time.sleep(0.01)
