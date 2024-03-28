from epochalyst.pipeline.ensemble import EnsemblePipeline


class BaseEnsemble(EnsemblePipeline):

    def concat(self, original_data, new_data, weight):
        return original_data + new_data * weight
