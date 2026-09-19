from .adapters import run_real_stage

class MaterializationAdapter:
    def run(self, paths, parents):
        return run_real_stage(paths, "LLMMaterialize", parents)
