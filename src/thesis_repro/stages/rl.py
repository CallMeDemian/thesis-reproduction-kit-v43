from .adapters import run_heavy_gate, run_real_stage

class RLAdapter:
    def run(self, paths, stage, parents):
        return run_real_stage(paths, stage, parents)
    def heavy_gate(self, paths, parents):
        return run_heavy_gate(paths, parents)
