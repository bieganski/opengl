from numpy.lib.twodim_base import tri
from model_parser import ModelParser
from drawer import draw_pixels, bresenham, triangle_bbox_iterate

from pathlib import Path
from operator import itemgetter
from itertools import combinations, chain
from dataclasses import dataclass
from typing import Callable, List, Tuple

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

path = Path("model.obj")

class Solution():

    @staticmethod
    def lab1(): # bresenham
        vs = ModelParser.get_vertices(path)
        fs = ModelParser.get_faces(path)

        print(f"len vertices = {len(vs)}")
        print(f"len faces = {len(fs)}")

        w, h = 400, 400
        pixels = []
        for f in fs:
            fcoords3 = itemgetter(*f)(vs)
            # we don't care about Z axis for now.
            coords = list(map(lambda abc: (int((abc[0] + 1.0) * w/2) - 1, int((abc[1] + 1.0) * h/2) - 1), fcoords3))
            lines = combinations(coords, 2)
            for c1, c2 in lines:        
                res = list(bresenham(*c1, *c2))
                assert all(x < max(w,h) for x in list(chain(*res)))
                pixels.extend(res)

        draw_pixels(pixels, w, h)

    # https://stackoverflow.com/a/40360416
    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            norm = np.finfo(float).eps
        return v / norm


    # returns normalized value - from interval [-1.0; 1.0]
    @staticmethod
    def calculate_color_intensity(triangle_fcoords):
        triangle_fcoords = [list(x) for x in triangle_fcoords]
        a, b, c = map(np.array, triangle_fcoords)
        ab, ac = b - a, c - a
        v = np.cross(ab, ac)
        v = Solution.normalize(v)
        light = np.array([0,1,0])
        intensity = np.dot(light, v)
        mmax = sum(abs(light))
        fr, to = [-mmax, mmax], [-1., 1.]
        intensity = np.interp(intensity, fr, to)
        return intensity
        
    @staticmethod
    def calculate_color_back_face_culling(all_fcoords3, _, __):
        for triangle in all_fcoords3:
            intensity = __class__.calculate_color_intensity(triangle)
            if intensity <= 0:
                yield None
            else:
                fr, to = [0., 1.], [0., 255.] # no ABS
                intensity = np.interp(intensity, fr, to)
                yield int(intensity)

    @staticmethod
    def calculate_color_zbuffer(all_fcoords3, w, h):
        for triangle in all_fcoords3:
            intensity = __class__.calculate_color_intensity(triangle)
            fr, to = [-1., 1.], [0., 255.] # ABS
            intensity = np.interp(intensity, fr, to)
            yield int(intensity)

    # * finds bounding box and calculates barycentric coordinates
    # * returns pixel if and only if all it's barycentric coords are positive
    @staticmethod
    def fill_full_triangle(vs : List[Tuple[int, int]]):
        f = lambda x: all(x >= 0)
        return triangle_bbox_iterate(vs, f)

    # * finds bounding box and calculates barycentric coordinates
    # * draws pixel if zcoords are ok XXX
    @staticmethod
    def fill_triangle_zbuffer(vs : List[Tuple[int, int]]):

        @dataclass
        class State():
            w : int
            h : int
            zbuffer : np.ndarray = None
            def __post_init__(self):
                self.zbuffer = np.empty(shape=(self.w, self.h))

        state = State(400, 400) # XXX use w,h

        def f(barycentric_coords):
            nonlocal state # capture state (preserved for all invocations)
            return True
        
        return triangle_bbox_iterate(vs, f)

    @staticmethod
    def shade_polygons(render_function : Callable, fill_function : Callable):
        vs = ModelParser.get_vertices(path)
        fs = ModelParser.get_faces(path)
            
        w, h = 400, 400
        im = Image.new("L", (w, h))

        lmap = lambda f, x : list(map(f,x))
        
        fcoords3 = [itemgetter(*f)(vs) for f in fs]
        def f3_to_i2(f3):
            # discards Z axis
            f = lambda abc: (int((abc[0] + 1.0) * w/2) - 1, int((abc[1] + 1.0) * h/2) - 1)
            return lmap(f, f3)
        icoords2 = lmap(f3_to_i2, fcoords3)

        render_fn = render_function(fcoords3, w, h)
        for color, triangle in tqdm(zip(render_fn, icoords2)):
            if color is None:
                continue
            pixels = fill_function(triangle)
            for p in pixels:
                im.putpixel(p, value=color)
        im = np.asarray(im)
        im = np.flip(im, 0)
        plt.imshow(im)
        plt.show()

    @staticmethod
    def lab2():
        __class__.shade_polygons(__class__.calculate_color_back_face_culling, __class__.fill_full_triangle)

    @staticmethod
    def lab3():
        __class__.shade_polygons(__class__.calculate_color_zbuffer, __class__.fill_triangle_zbuffer)
