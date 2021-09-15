from pathlib import Path

class ModelParser():
    @staticmethod
    def get_vertices(path : Path):
        lines = path.open().readlines()
        f = float
        return [list(map(f, x[2:].split(" "))) for x in lines if x.startswith("v ")]

    @staticmethod
    def get_faces(path : Path):
        lines = path.open().readlines()
        f = lambda x : int(x.split("/")[0]) - 1
        return [list(map(f, x[2:].split(" "))) for x in lines if x.startswith("f ")]