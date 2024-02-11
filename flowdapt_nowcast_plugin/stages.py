import numpy as np
import pandas as pd
import logging

from typing import Dict
from datetime import timezone
from flowdapt.lib.errors import ResourceNotFoundError
from flowdapt.builtins import utils
from flowdapt.compute.resources.workflow.context import get_run_context
from flowdapt.compute import object_store
from flowml.pipefill import PipeFill
from datasieve.pipeline import Pipeline
import datasieve.transforms as ds
from sklearn.preprocessing import MinMaxScaler
from flowdapt.compute.artifacts.misc import json_to_artifact, json_from_artifact
from flowdapt.compute.artifacts.dataset.dataframes import (
    dataframe_to_artifact,
    dataframe_from_artifact
)

from flowdapt_nowcast_plugin.api import download_data, get_city_grid
from flowdapt_nowcast_plugin.utils import get_package_datafile_path
import flowml.utils as mlutils

CLUSTER_MEMORY_ACTOR_NAME = "cluster_memory"

logger = logging.getLogger(__name__)


def create_parameterization_lists():
    """
    STAGE
    Creates a set of lists and publishes them to the object store so they
    are available for parameterization purposes in other stages/workflows
    """

    context = get_run_context()
    config = context.config
    data_config = context.config["data_config"]
    # first check if we created the lists before, otherwise return
    try:
        unique_cities = object_store.get(
            "unique_cities",
            artifact_only=True,
            load_artifact_hook=json_from_artifact(),
            namespace=config["study_identifier"]
        )

        if len(unique_cities) > 0:
            logger.info("Using existing city lists from artifacts.")
            return
    except:
        logger.info("No unique cities found in object_store, creating new lists.")

    nbrs = data_config["neighbors"]
    radius = data_config["radius"]

    neighbor_construct = get_city_grid(
        data_config["cities"],
        nbrs,
        radius,
        get_package_datafile_path("data/uscities.csv", "flowdapt_nowcast_plugin")
    )

    # build the iterable for our model comparisons
    city_model_iterable = pd.DataFrame()
    base_list = neighbor_construct[neighbor_construct["neighbor_num"] == 0].copy()
    for target in data_config["targets"]:
        base_list.loc[:, "target"] = target
        city_model_iterable = pd.concat([city_model_iterable, base_list], axis=0)

    # shuffle the city_model_iterable to better balance worker load
    city_model_iterable = city_model_iterable.sample(frac=1, random_state=12).reset_index(drop=True)
    city_model_iterable = city_model_iterable.to_dict(orient='records')

    # save the list of unique cities
    cities_list = neighbor_construct[
        neighbor_construct["neighbor_num"] == 0
    ].to_dict(orient='records')

    neighbors_dict = neighbor_construct.to_dict(orient='records')
    unique_list = neighbor_construct.drop_duplicates(subset=['city']).to_dict(orient="records")

    object_store.put(
        "city_model_iterable",
        city_model_iterable,
        artifact_only=True,
        save_artifact_hook=json_to_artifact(),
        namespace=config["study_identifier"]
    )
    object_store.put(
        "cities_list", cities_list, artifact_only=True,
        save_artifact_hook=json_to_artifact(),
        namespace=config["study_identifier"]
    )
    object_store.put(
        "neighbors_dict", neighbors_dict, artifact_only=True,
        save_artifact_hook=json_to_artifact(),
        namespace=config["study_identifier"]
    )
    object_store.put(
        "unique_cities", unique_list, artifact_only=True,
        save_artifact_hook=json_to_artifact(),
        namespace=config["study_identifier"]
    )

    return


# These stages only serve a purpose of preceding
# the parameterized stages.
def get_unique_city_iterable(*args):
    study_identifier = get_run_context().config["study_identifier"]
    return object_store.get(
        "unique_cities",
        artifact_only=True,
        load_artifact_hook=json_from_artifact(),
        namespace=study_identifier
    )


def get_target_city_iterable(*args):
    study_identifier = get_run_context().config["study_identifier"]
    return object_store.get(
        "cities_list",
        artifact_only=True,
        load_artifact_hook=json_from_artifact(),
        namespace=study_identifier
    )


def get_city_model_iterable(*args):

    # log the start of the workflow
    study_identifier = get_run_context().config["study_identifier"]
    return object_store.get(
        "city_model_iterable",
        artifact_only=True,
        load_artifact_hook=json_from_artifact(),
        namespace=study_identifier
    )


# stage that is parameterized across the first argument (row).
def get_and_publish_data(row: Dict):
    context = get_run_context()
    logger.info(f"Updating data for {row['city']}")

    config = context.config
    data_config = config["data_config"]

    md = {
        "city": row["city"],
        "latitude": row["lat"],
        "longitude": row["lng"],
        "num_days": data_config["n_days"]
    }

    object_name = utils.artifact_name_from_dict({"city": row["city"]})
    # TODO: We should have a single error such as ResourceNotFound the object store throws
    # that we can catch here instead of having to catch two different errors. Not needed
    # in this case since we know we're only using Artifacts right here but for others it will be.
    try:
        df = object_store.get(
            f"{object_name}_raw",
            artifact_only=True,
            load_artifact_hook=dataframe_from_artifact(format="parquet"),
            namespace=context.config["study_identifier"]
        )
        logger.info(f"Found {object_name}_raw in artifact with len {len(df.index)}")
    except FileNotFoundError:
        df = pd.DataFrame()

    logger.info(f"DLing for {row['city']}")

    if len(df.index) == 0:
        n_bars = config["extras"]["num_points"]
    else:
        if len(df) < config["extras"]["num_points"]:
            logger.info(f"df is {len(df)} but needs {config['extras']['num_points']}")
            n_bars = config["extras"]["num_points"]
        else:
            n_bars = utils.get_data_gap(df, timezone.utc)

    if n_bars > 0:
        logger.info(f"Data n_bars {n_bars} for {row['city']}, downloading.")
        df_inc, df_inc_fc = download_data(md, n_bars)

        if len(df.index) > 0:
            df = utils.merge_dataframes(df, df_inc, prefer_new=False)
        else:
            df = df_inc

    df = mlutils.reduce_dataframe_footprint(df)
    # Save the raw dataframe and forecast dataframes as artifacts on disk
    object_store.put(
        f"{object_name}_raw", df,
        artifact_only=True,
        save_artifact_hook=dataframe_to_artifact(format="parquet"),
        namespace=context.config["study_identifier"]
    )

    df = df.tail(config["extras"]["num_points"])
    df = feature_engineering(row, df)

    # Save the features dataframe in the Object Store
    object_store.put(
        f"{object_name}_features",
        df,
        namespace=context.config["study_identifier"]
    )

    return


def feature_engineering(row: dict, df):

    # prepend columns with % so they are recognized as features
    # embed city in feature name so they can be distinguished in
    # aggregated dfs
    df.columns = [f'%-{row["city"]}-{i}' if i != 'date'
                  else i for i in df.columns]

    # add hour of day
    df = df.copy()  # Make a copy of the original dataframe
    hour = df["date"].dt.hour
    hour_norm = 2 * np.pi * \
        hour / hour.max()
    df[f'%-{row["city"]}-hour_of_day_cos'] = np.cos(hour)
    df[f'%-{row["city"]}-hour_of_day_sin'] = np.sin(hour_norm)
    df.loc[:, f'%-{row["city"]}-hour_of_day_cos'] = np.cos(hour)
    df.loc[:, f'%-{row["city"]}-hour_of_day_sin'] = np.sin(hour_norm)

    return df


def construct_dataframes(row: dict):
    """
    STAGE
    Gets data from datastore (hopefully already published data)
    and creates a single dataframe concatenated with
    all requested datasources
    """
    context = get_run_context()

    config = context.config
    data_config = config["data_config"]

    object_name = utils.artifact_name_from_dict({"city": row["city"]})
    df = object_store.get(f"{object_name}_features", namespace=config["study_identifier"])
    nbr_list = object_store.get(
        "neighbors_dict", artifact_only=True,
        load_artifact_hook=json_from_artifact(),
        namespace=config["study_identifier"]
    )

    # load the cities artifact to dataframe
    df_cities = pd.DataFrame.from_dict(nbr_list)

    # get the city and city group for the current source
    city = df_cities[(df_cities['city'] == row['city']) & (df_cities['neighbor_num'] == 0)]
    grp = city["city_group"].values[0]
    group = df_cities[df_cities["city_group"] == grp]

    for _, city in group.iterrows():
        if city["city"] == row["city"]:
            # avoid grabbing the current source twice
            continue

        aux_name = utils.artifact_name_from_dict({"city": city["city"]})

        # pull auxillary features from cluster memory
        try:
            df_aux = object_store.get(
                f"{aux_name}_features",
                namespace=config["study_identifier"]
            )
        except ResourceNotFoundError:
            logger.error(f"No dataset available for {aux_name}, skipping.")
            continue

        df = pd.merge(df, df_aux, how='left', on='date')

    df = set_weather_targets(row, df, data_config["targets"])

    object_store.put(
        f"{object_name}_prepped_df",
        df,
        namespace=config["study_identifier"]
    )

    return


def set_weather_targets(row, raw_df, targets):

    target_base = []
    for target in targets:
        target_base.append(f"%-{row['city']}-{target}")

    target_list = []

    def add_target(df, shift):
        df_shift = df[target_base] - df[target_base].shift(shift)
        df_shift = df_shift.add_suffix(f"-{shift}hr")
        df_shift = df_shift.add_prefix("&-")
        target_list.append(df_shift.columns)
        df = pd.concat([df, df_shift], axis=1)
        return df

    raw_df = add_target(raw_df, -1)
    raw_df = add_target(raw_df, -2)
    raw_df = add_target(raw_df, -3)
    raw_df = add_target(raw_df, -4)
    raw_df = add_target(raw_df, -5)
    raw_df = add_target(raw_df, -6)

    return raw_df


def train_pipeline(row: dict):
    """
    Parameterized Stage
    Training pipeline
    """
    context = get_run_context()

    config = context.config
    data_config = config["data_config"]

    object_name = utils.artifact_name_from_dict({"city": row["city"]})
    pipefill_name = f"{object_name}_target-{row['target']}"

    # Load the pipefill from the object store
    try:
        pf: PipeFill = object_store.get(
            pipefill_name,
            artifact_only=config["extras"]["artifact_only"],
            load_artifact_hook=PipeFill.from_artifact(),
            namespace=config["study_identifier"]
        )
    except (KeyError, FileNotFoundError):
        pf = PipeFill(
            name_str=pipefill_name,
            namespace=config["study_identifier"],
            model_str=f"flowml.{data_config['model']}",
            model_train_params=config["model_train_parameters"],
            data_split_params=config["extras"]["data_split_parameters"],
            extras=config["extras"]
        )

    raw_df = object_store.get(f"{object_name}_prepped_df", namespace=config["study_identifier"])
    feature_list = utils.find_features(raw_df)
    raw_df = utils.shift_and_add_features(
        raw_df, config["model_train_parameters"]["lookback"],
        feature_list
    )

    raw_df = utils.remove_none_columns(raw_df, threshold=40)
    raw_df = utils.remove_rows_with_nans(raw_df)

    logger.debug(f"Analyzed df for {row['city']}, {raw_df}")
    features, labels = utils.extract_features_and_labels(raw_df)
    labels = labels.filter(regex=row['target'])
    pf.feature_list = features.columns
    pf.label_list = labels.columns
    if features.empty or labels.empty:
        logger.warning(f"No features or labels found for {row['city']}")
        return {}

    # helper function to automatically split the train and test datasets inside the
    # pipefill and create the eval_set
    data_params = config["extras"]["data_split_parameters"]
    w_factor = config["extras"]["weight_factor"]
    X, X_test, y, y_test, w, w_test = mlutils.make_train_test_datasets(features,
                                                                       labels,
                                                                       w_factor,
                                                                       data_params)

    # data processing pipelines
    pf.feature_pipeline = Pipeline([
        ("raw_scaler", ds.SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))),
        ("detect_constants", ds.VarianceThreshold(threshold=0)),
    ])

    # fit the feature pipeline to the features and transform them in one call
    X, y, w = pf.feature_pipeline.fit_transform(X, y, w)
    # transform the test set using the fitted pipeline
    X_test, y_test, w_test = pf.feature_pipeline.transform(X_test, y_test, w_test)

    # the labels require a separate pipeline (the objects are fit in a the label parameter
    # space.)
    pf.target_pipeline = Pipeline([
        ("scaler", ds.SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))),
    ])

    y, _, _ = pf.target_pipeline.fit_transform(y)
    y_test, _, _ = pf.target_pipeline.transform(y_test)

    eval_set = [(X_test, y_test)]

    logger.info(f"About to train on city {row['city']}")
    pf.model.fit(X, y, eval_set=eval_set)

    logger.info(f"Model trained {pf.model}")

    # set important info to keep for later
    pf.set_trained_timestamp()
    pf.metadata["num_points"] = len(X.index)
    pf.best_loss = pf.model.best_loss

    # Add to cluster memory for quicker access
    object_store.put(
        pipefill_name,
        pf,
        artifact_only=config["extras"]["artifact_only"],
        save_artifact_hook=pf.to_artifact(),
        namespace=config["study_identifier"]
    )

    return


def predict_pipeline(row: dict):
    """
    Parameterized Stage
    Predict stage
    """
    context = get_run_context()
    config = context.config

    object_name = utils.artifact_name_from_dict({"city": row["city"]})
    pipefill_name = f"{object_name}_target-{row['target']}"

    num_points = config["model_train_parameters"]["lookback"]

    # Load the pipefill from the object store, if it's in cluster memory then
    # it'll get loaded from there, otherwise just get from the Artifact
    pf = object_store.get(
        pipefill_name,
        artifact_only=config["extras"]["artifact_only"],
        load_artifact_hook=PipeFill.from_artifact(),
        namespace=config["study_identifier"]
    )

    raw_df = object_store.get(f"{object_name}_prepped_df", namespace=config["study_identifier"])
    raw_df = raw_df.tail(num_points)

    features, _ = utils.extract_features_and_labels(raw_df)
    features = features.filter(items=pf.feature_list, axis=1)
    features = utils.remove_rows_with_nans(features)

    target = f"%-{row['city']}-{row['target']}"
    ground_truth = features[target].iloc[-1:]

    features = utils.shift_and_add_features(
        features, config["model_train_parameters"]["lookback"], features.columns)
    features = features.filter(items=pf.feature_list, axis=1)
    features = features.iloc[-1, :]
    features = features.values.reshape(1, -1)

    features, _, _ = pf.feature_pipeline.transform(features)
    preds = pf.model.predict(features)

    preds, _, _ = pf.target_pipeline.inverse_transform(preds.reshape(1, -1))
    # convert back to actual values if the targets were differences
    for col in preds.columns:
        preds[col] = ground_truth.values - preds[col]

    preds = preds.to_numpy().squeeze()
    logger.debug(f"preds for {row['city']} \n {preds} ")

    dates = raw_df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    date = list(dates)[-1]

    # build_historic_predictions(row, preds, pf.label_list, ground_truth, target, date)
    logger.warning(f"preds: {preds}")

    # convert preds numpyarray to list
    return {
        "preds": {"row": row, "preds": preds.tolist(), "date": date, "labels": list(pf.label_list),
                  "ground_truth": ground_truth.values.tolist(), "target": [target]}
    }
