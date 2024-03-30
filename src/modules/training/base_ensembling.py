from epochalyst.pipeline.ensemble import EnsemblePipeline


class BaseEnsemble(EnsemblePipeline):
    """BaseEnsemble pipeline for other ensembling methods override concat"""
    # def concat(self, original_data, new_data, weight):
    #     return original_data + new_data * weight
