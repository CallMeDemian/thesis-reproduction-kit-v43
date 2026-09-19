from .adapters import run_real_stage

class LLMAdapter:
    def run(self, paths, stage, parents, execute_llm=False):
        return run_real_stage(paths, stage, parents, execute_llm=execute_llm)
