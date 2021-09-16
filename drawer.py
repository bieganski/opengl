#!/usr/bin/env python3

from typing import List, Tuple
import numpy as np

# from drawer import bresenham

def draw_pixels(pixels : List[Tuple[int, int]], w=640, h=480, color=0xff):
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    im = Image.new("L", (w, h))
    for p in pixels:
        im.putpixel(p, value=color)
    # im.show()
    plt.imshow(np.asarray(im))
    plt.show()

def cross(v1, v2):
    a = v1[1] * v2[2] - v1[2] * v2[1]
    b = v1[2] * v2[0] - v1[0] * v2[2]
    c = v1[0] * v2[1] - v1[1] * v2[0]
    return a,b,c

# * finds bounding box
# * calculates barycentric coordinates
# * draws pixel only if all it's barycentric coords are positive
def fill_triangle(vs : List[Tuple[int, int]], w: int, h: int):
    assert len(vs) == 3
    from itertools import product

    ws, hs = zip(*vs)
    bbox_top, bbox_bot = max(hs), min(hs)
    bbox_r, bbox_l = max(ws), min(ws)

    points = list(product(
        range(bbox_l, bbox_r + 1),
        range(bbox_bot, bbox_top + 1),
    ))
    assert len(points) == (bbox_top - bbox_bot + 1) * (bbox_r - bbox_l + 1)
    
    points = np.array(points)
    vs = np.array(vs) # for vectorized + and - ops
    
    # TODO
    # why the hell I had to use BA and CA instead of AC, CA to make it work?
    a = vs[0]
    ab = a - vs[1]
    ac = a - vs[2]

    pixels = []
    for p in points:
        pa = p - a
        v1, v2 = np.array([ac, ab, pa]).T.astype(float)
        x, y, z = np.cross(v1, v2).astype(float)
        if z == 0:
            continue
        barycentric = np.array([
            1.0 - (x + y) / z, 
            x / z,
            y / z
        ])
        if all(barycentric >= 0):
            pixels.append(p)
    return pixels     
    # draw_pixels(pixels, w, h, color=color)


def bresenham(x0, y0, x1, y1) -> Tuple[int, int]:
    return list(_bresenham(x0, y0, x1, y1))

def _bresenham(x0, y0, x1, y1) -> Tuple[int, int]:
    dx = abs(x0 - x1)
    dy = abs(y0 - y1)
    err = dx - dy
    while (x0, y0) != (x1, y1):
        yield (x0, y0)
        e2 = 2 * err
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        
        if e2 >= -dy:
            # go horizontal
            x0 += sx
            err -= dy
        if e2 <= dx:
            # go vertical
            y0 += sy
            err += dx

def draw_line(x0, y0, x1, y1, w=640, h=480):
    pixels = bresenham(x0, y0, x1, y1)
    draw_pixels(pixels, w, h)