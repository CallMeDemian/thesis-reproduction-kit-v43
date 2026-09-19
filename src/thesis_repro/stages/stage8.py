from .adapters import run_real_stage

class Stage8Adapter:
    def run(self, paths, parents):
        return run_real_stage(paths, "Stage8", parents)
