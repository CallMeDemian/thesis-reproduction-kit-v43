from .adapters import run_real_stage

class C3EAdapter:
    def run(self, paths, parents):
        return run_real_stage(paths, "C3E", parents)
