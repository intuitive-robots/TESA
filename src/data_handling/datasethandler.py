from src.data_handling import datasets, rie_datasets
from src.util.util import ds_base_embedding


def rie_ds(raw_data, config, forTrainingOnly=False):
    dataset_name = config["dataset"]
    initial_features = config["initial_features"]

    if config.get("relation_is") == "node":
        print("(datasethandler: relations are Nodes)")
        return datasets.RelIsNodeUnifiedJsonDataset(
            ds_base_embedding=ds_base_embedding(config),
            raw_data=raw_data,
            ds_name=dataset_name,
            initial_features=initial_features,
            for_training_only=forTrainingOnly,
        )
    return rie_datasets.RelIsEdgeUnifiedJsonDataset(
        ds_base_embedding=ds_base_embedding(config),
        raw_data=raw_data,
        ds_name=dataset_name,
        initial_features=initial_features,
        for_training_only=forTrainingOnly,
    )
