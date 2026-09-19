from .adapters import run_real_stage

class Stage9Adapter:
    def run(self, paths, parents):
        return run_real_stage(paths, "Stage9", parents)
