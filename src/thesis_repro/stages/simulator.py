from .adapters import run_real_stage

class SimulatorAdapter:
    name = "Simulator"
    def run(self, paths, parents):
        return run_real_stage(paths, self.name, parents)
