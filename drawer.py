#!/usr/bin/env python3

from typing import List, Tuple
from profiler import Profiler
# from drawer import bresenham

def draw_pixels(pixels : List[Tuple[int, int]], w=640, h=480):
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    im = Image.new("L", (w, h))
    for p in pixels:
        im.putpixel(p, 255)
    # im.show()
    plt.imshow(np.asarray(im))
    plt.show()


def fill_triangle(vs : List[Tuple[int, int]], w: int, h: int):
    assert len(vs) == 3

    # sort vertices descending by height
    descending = lambda lst2 : sorted(lst2, key=lambda x : x[1], reverse=True)
  
    def getl2r2(top, mid, bot):
        pass
        l2 = bresenham(*bot, *top)
        l2.reverse()
        assert descending(l2) == l2
        r2_1 = bresenham(*bot, *mid)
        r2_1.reverse()
        assert descending(r2_1) == r2_1
        r2_2 = bresenham(*mid, *top)
        r2_2.reverse()
        assert descending(r2_2) == r2_2
        r2 = r2_1 + r2_2
        assert len(l2) == len(r2), f"{len(l2)}xx{len(r2)}"

        while len(l2):
            yield l2.pop(), r2.pop()

    fill = []
    for l2, r2 in getl2r2(*descending(vs)):
        fill.extend(bresenham(*l2, *r2))
    draw_pixels(fill, w, h)


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