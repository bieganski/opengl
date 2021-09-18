from numpy.lib.twodim_base import tri
from model_parser import ModelParser
from drawer import draw_pixels, bresenham, fill_triangle

from pathlib import Path
from operator import itemgetter
from itertools import combinations, chain

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

    @staticmethod
    def lab2():
        vs = ModelParser.get_vertices(path)
        fs = ModelParser.get_faces(path)

        def calculate_color(triangle):

            # https://stackoverflow.com/a/40360416
            def normalize(v):
                norm=np.linalg.norm(v, ord=1)
                if norm==0:
                    norm=np.finfo(float).eps
                return v/norm
            triangle = [list(x) for x in triangle]
            a, b, c = map(np.array, triangle)
            ab, ac = b - a, c - a
            v = np.cross(ab, ac)
            v = normalize(v)
            light = np.array([0.5,1,0])
            intensity = np.dot(light, v)
            if intensity <= 0:
                return None
            from numpy import interp
            mmax  = sum(abs(light))
            fr, to = [-mmax, mmax], [0., 255.]
            intensity = interp(intensity, fr, to)
            return int(intensity)
            
        w, h = 400, 400
        im = Image.new("L", (w, h))
        for f in tqdm(fs):
            fcoords3 = itemgetter(*f)(vs)
            # we don't care about Z axis for now.
            triangle = list(map(lambda abc: (int((abc[0] + 1.0) * w/2) - 1, int((abc[1] + 1.0) * h/2) - 1), fcoords3))
            color = calculate_color(fcoords3)
            if color is None:
                continue
            pixels = fill_triangle(triangle, w, h)
            for p in pixels:
                im.putpixel(p, value=color)
        plt.imshow(np.asarray(im))
        plt.show()