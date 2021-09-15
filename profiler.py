
class Profiler():
    def __enter__(self):
        import cProfile
        self.pr = cProfile.Profile()
        self.pr.enable()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.pr.disable()
        from pstats import SortKey, Stats
        ps = Stats(self.pr).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats()
