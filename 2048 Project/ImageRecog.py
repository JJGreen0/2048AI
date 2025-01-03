from PIL import Image, ImageGrab
import pytesseract
import time

grid = [[0 for _ in range(4)] for _ in range(4)]

# Load the processed image
def GiveState(image):

    tile_width = image.width // 4
    tile_height = image.height // 4

    for i in range(4):
        for j in range(4):
            # Define the bounds of the tile
            left = j * tile_width
            top = i * tile_height
            right = left + tile_width
            bottom = top + tile_height
            # Crop the tile from the image
            tile = image.crop((left, top, right, bottom))
            # Use pytesseract to do OCR on the cropped tile
            text = pytesseract.image_to_string(tile, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789')
            # If the OCR result is a digit, place it in the grid, otherwise assume the tile is empty
            grid[i][j] = int(text.strip()) if text.strip().isdigit() else 0
    
    return grid

def TakeImage():
    image = ImageGrab.grab(bbox =(700,314,1204,816))
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((image.width // 4, image.height // 4))
    image.show()
    return image

loop_time = time.time()

while True:
    image = TakeImage()

    #grid = GiveState(image)

    print('FPS {}'.format(1/(time.time() - loop_time)))
    loop_time = time.time()

