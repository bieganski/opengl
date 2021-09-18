from numpy.lib.twodim_base import tri
from model_parser import ModelParser
from drawer import draw_pixels, bresenham, fill_triangle

from pathlib import Path
from operator import itemgetter
from itertools import combinations, chain
from dataclasses import dataclass

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

    @staticmethod
    def calculate_color_back_face_culling(all_fcoords3, _, __):
        for triangle in all_fcoords3:
            triangle = [list(x) for x in triangle]
            a, b, c = map(np.array, triangle)
            ab, ac = b - a, c - a
            v = np.cross(ab, ac)
            v = Solution.normalize(v)
            light = np.array([0,1,0])
            intensity = np.dot(light, v)
            # yield must be the last instruction in a loop to work properly.
            if intensity <= 0:
                yield None
            else:
                mmax = sum(abs(light))
                fr, to = [-mmax, mmax], [0., 255.]
                intensity = np.interp(intensity, fr, to)
                yield int(intensity)

    @staticmethod
    def calculate_color_zbuffer(all_fcoords3, w, h):
        
        @dataclass
        class State():
            w : int
            h : int
            _zbuffer : np.ndarray

            def __post_init__(self):
                self._zbuffer = np.empty(shape=(self.w, h))
        
        a = State()
        raise NotImplementedError()

    @staticmethod
    def shade_polygons(render_function):
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
            pixels = fill_triangle(triangle, w, h)
            for p in pixels:
                im.putpixel(p, value=color)
        im = np.asarray(im)
        im = np.flip(im, 0)
        plt.imshow(im)
        plt.show()

    @staticmethod
    def lab2():
        Solution.shade_polygons(Solution.calculate_color_back_face_culling)

    @staticmethod
    def lab3():
        Solution.shade_polygons(Solution.calculate_color_zbuffer)
