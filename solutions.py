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

path = Path("model/model.obj")

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
    def calculate_color_intensity_norm(triangle_fcoords):
        triangle_fcoords = [list(x) for x in triangle_fcoords]
        a, b, c = map(np.array, triangle_fcoords)
        ab, ac = b - a, c - a
        v = np.cross(ab, ac)
        v = Solution.normalize(v)
        light = np.array([1,1,1])
        intensity = np.dot(light, v)
        mmax = sum(abs(light))
        fr, to = [-mmax, mmax], [-1., 1.]
        intensity = np.interp(intensity, fr, to)
        return intensity
        
    @staticmethod
    def calculate_color_back_face_culling(all_fcoords3):
        for triangle in all_fcoords3:
            intensity = __class__.calculate_color_intensity_norm(triangle)
            if intensity <= 0:
                yield None
            else:
                fr, to = [0., 1.], [0., 255.]
                intensity = np.interp(intensity, fr, to)
                yield int(intensity)

    @staticmethod
    def calculate_color_zbuffer(all_fcoords3):
        for triangle in all_fcoords3:
            intensity = __class__.calculate_color_intensity_norm(triangle)
            intensity = abs(intensity)
            fr, to = [0., 1.], [0., 255.]
            intensity = np.interp(intensity, fr, to)
            yield int(intensity)

    # * finds bounding box and calculates barycentric coordinates
    # * returns pixel if and only if all it's barycentric coords are positive
    @staticmethod
    def fill_full_triangle(vs : List[Tuple[int, int, int]]):
        f = lambda _, barycentric_coords, __: not any(barycentric_coords < 0)
        return triangle_bbox_iterate(vs, f)

    @dataclass
    class State():
        w : int
        h : int
        zbuffer : np.ndarray = None
        def __post_init__(self):
            self.zbuffer = np.empty(shape=(self.w, self.h), dtype=float)
            self.zbuffer.fill(np.finfo(float).min)

    state = State(400, 400) # XXX use w,h
    
    @staticmethod
    def f(vs, barycentric_coords, euclidean_coords):
        # nonlocal __class__.state # capture state (preserved for all invocations)
        if any(barycentric_coords < 0):
            return False # not inside triangle
        z_axis = [c[2] for c in vs]
        z = sum([w * c for w, c in zip(barycentric_coords, z_axis)])
        x, y = euclidean_coords
        state = __class__.state
        if state.zbuffer[x, y] < z:
            state.zbuffer[x, y] = z
            return True
        else:
            return False # inside triangle but invsible - tucked behind

    # * finds bounding box and calculates barycentric coordinates
    # * draws pixel only if it will be visible, according to current state (zbuffer)
    @staticmethod
    def fill_triangle_zbuffer(vs : List[Tuple[int, int, int]]):
        return triangle_bbox_iterate(vs, __class__.f)


    @staticmethod
    def shade_polygons(render_function : Callable, fill_function : Callable):
        vs = ModelParser.get_vertices(path)
        vts = ModelParser.get_texture_vertices(path)
        fs = ModelParser.get_faces(path)
        ts = ModelParser.get_textures(path)
            
        w, h = 400, 400
        im = Image.new("L", (w, h))

        lmap = lambda f, x : list(map(f,x))
        
        def f3_to_i3(f3):
            f = lambda abc: (int((abc[0] + 1.0) * w/2) - 1, int((abc[1] + 1.0) * h/2) - 1, int((abc[2] + 1.0) * w/2) - 1)
            return lmap(f, f3)
        
        fcoords3 = [itemgetter(*f)(vs) for f in fs]
        tvcoords3 = [itemgetter(*f)(vts) for f in ts]
        icoords2 = lmap(f3_to_i3, fcoords3)

        # raise ValueError(tvcoords3[0])

        render_fn = render_function(fcoords3)
        for color_intensity, triangle, texture in tqdm(zip(render_fn, icoords2, ts)):
            if color_intensity is None:
                continue
            color = color_intensity # XXX
            for pixel in fill_function(triangle):
                im.putpixel(pixel, value=color)
        
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
