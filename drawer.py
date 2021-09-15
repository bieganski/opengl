#!/usr/bin/env python3

from typing import List, Tuple
from profiler import Profiler

def draw_pixels(pixels : List[Tuple[int, int]], w=640, h=480):
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    im = Image.new("L", (w, h))
    for p in pixels:
        im.putpixel(p, 255)
    # im.show()
    plt.imshow(np.asarray(im))
    # plt.show()


def bresenham(x0, y0, x1, y1) -> Tuple[int, int]:
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
    pixels = list(bresenham(x0, y0, x1, y1))
    draw_pixels(pixels, w, h)

with Profiler():
    draw_line(10, 10, 200, 400)