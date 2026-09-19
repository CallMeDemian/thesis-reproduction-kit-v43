from .adapters import run_real_stage

class OracleAdapter:
    name = "Oracle"
    def run(self, paths, parents):
        return run_real_stage(paths, self.name, parents)
