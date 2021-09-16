from model_parser import ModelParser
from pathlib import Path
from drawer import draw_pixels, bresenham, fill_triangle

path = Path("model.obj")

class Solution():

    @staticmethod
    def lab1(): # bresenham
        vs = ModelParser.get_vertices(path)
        fs = ModelParser.get_faces(path)

        print(f"len vertices = {len(vs)}")
        print(f"len faces = {len(fs)}")

        from operator import itemgetter
        from itertools import combinations, chain

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

        fill_triangle([(3,4), (100,50), (300, 300)], 400, 400)