from pathlib import Path

class ModelParser():
    @staticmethod
    def get_vertices(path : Path):
        lines = path.open().readlines()
        f = float
        return [list(map(f, x.split()[1:])) for x in lines if x.startswith("v ")]

    @staticmethod
    def get_texture_vertices(path : Path):
        lines = path.open().readlines()
        f = float
        return [list(map(f, x.split()[1:])) for x in lines if x.startswith("vt ")]

    @staticmethod
    def get_faces(path : Path):
        lines = path.open().readlines()
        f = lambda x : int(x.split("/")[0]) - 1
        return [list(map(f, x.split()[1:])) for x in lines if x.startswith("f ")]

    @staticmethod
    def get_textures(path : Path):
        lines = path.open().readlines()
        f = lambda x : int(x.split("/")[1]) - 1
        return [list(map(f, x.split()[1:])) for x in lines if x.startswith("f ")]