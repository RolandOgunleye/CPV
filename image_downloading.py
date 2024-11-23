import cv2
import requests
import numpy as np
import threading
import os  # For directory path handling

def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y

def download_tile(url, headers, channels):
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to download tile: {url} - Status code: {response.status_code}")
        return None
    
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Failed to decode image from tile: {url}")
    else:
        print(f"Successfully downloaded and decoded tile: {url}")
    
    return img

def download_image(lat1: float, lon1: float, lat2: float, lon2: float,
                   zoom: int, url: str, headers: dict, tile_size: int = 256, channels: int = 3, save_dir: str = "") -> None:
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    # Download tiles in the specified range
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile_url = url.format(x=tile_x, y=tile_y, z=zoom)
            tile_img = download_tile(tile_url, headers, channels)
            if tile_img is not None:
                # Save the downloaded tile
                tile_name = f'tile_{tile_x}_{tile_y}.png'
                tile_path = os.path.join(save_dir, tile_name)
                cv2.imwrite(tile_path, tile_img)
                print(f'Saved tile: {tile_path}')
            else:
                print(f'Failed to download tile: {tile_url}')

