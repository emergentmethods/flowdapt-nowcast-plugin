import pytest
import logging

from flowdapt.compute.resources.workflow.execute import execute_workflow
# from flowdapt.compute.executor.ray import RayExecutor
from flowdapt.compute.executor.local import LocalExecutor

from flowdapt_nowcast_plugin.stages import (
    get_city_model_iterable,
    get_target_city_iterable,
    get_unique_city_iterable,
    create_parameterization_lists,
    get_and_publish_data,
    construct_dataframes,
    train_pipeline,
    predict_pipeline,
)

logger = logging.getLogger(__name__)

TESTING_NAMESPACE = "testing"

# Use a Session scoped Executor so any data persisted to the object store
# lives for the entire test run
@pytest.fixture(scope="session")
async def executor():
    try:
        executor = LocalExecutor(use_processes=False)  # RayExecutor(gpus=1)
        logger.info("Starting Executor")
        await executor.start()
        yield executor
    finally:
        logger.info("Closing Executor")
        await executor.close()

@pytest.fixture
def workflow_config():
    return {
        "study_identifier": "full_uscities",
        "model_train_parameters": {
            "n_jobs": 4,
            "n_estimators": 100,
            "verbosity": 1,
            "epochs": 2,
            "batch_size": 5,
            "lookback": 5,
            "shuffle": False,
        },
        "data_config": {
            "n_days": 20,
            "neighbors": 2,
            "radius": 150,
            "prediction_points": 1,
            "target_horizon": 6,
            "city_data_path": "flowdapt_nowcast_plugin/data/uscities.csv",
            "models": "XGBoostRegressor",
            "cities": ['Los Angeles'],
            "targets": ["temperature_2m"]
        },
        "extras": {
            "weight_factor": 0.9,
            "di_threshold": 5.0,
            "num_points": 450,
            "artifact_only": False,
            "data_split_parameters": {
                "test_size": 0.05,
                "shuffle": False,
            },
        },
        # We need this here for testing so we aren't actually writing any files
        "storage": {
            "protocol": "memory",
            "base_path": "test_data",
        }
    }


@pytest.fixture
def create_features_workflow(workflow_config):
    return {
        "name": "openmeteo_create_features",
        "config": workflow_config,
        "stages": [
            {
                "name": "create_parameterization_lists",
                "target": create_parameterization_lists,
            },
            {
                "name": "get_unique_city_iterable",
                "target": get_unique_city_iterable,
                "depends_on": ["create_parameterization_lists"]
            },
            {
                "name": "get_and_publish_data",
                "target": get_and_publish_data,
                "depends_on": ["get_unique_city_iterable"],
                "type": "parameterized"
            },
            {
                "name": "get_target_city_iterable",
                "target": get_target_city_iterable,
                "depends_on": ["get_and_publish_data"],
            },
            {
                "name": "construct_dataframes",
                "target": construct_dataframes,
                "depends_on": ["get_target_city_iterable"],
                "type": "parameterized"
            }
        ]
    }


@pytest.fixture
def train_pipeline_workflow(workflow_config):
    return {
        "name": "openmeteo_train_pipeline",
        "config": workflow_config,
        "stages": [
            {
                "name": "get_city_model_iterable",
                "target": get_city_model_iterable,
            },
            {
                "name": "train_pipeline",
                "target": train_pipeline,
                "depends_on": ["get_city_model_iterable"],
                "type": "parameterized",
                "resources": {
                    "cpus": 1
                }
            },
        ]
    }

@pytest.fixture
def predict_pipeline_workflow(workflow_config):
    return {
        "name": "openmeteo_predict_pipeline",
        "config": workflow_config,
        "stages": [
            {
                "name": "get_city_model_iterable",
                "target": get_city_model_iterable,
            },
            {
                "name": "predict_pipeline",
                "target": predict_pipeline,
                "depends_on": ["get_city_model_iterable"],
                "type": "parameterized",
                "resources": {
                    "cpus": 1
                }
            },
        ]
    }


async def test_create_features(create_features_workflow, executor):
    # We set return_result to True so any errors that are raised in the stage
    # are bubbled up here so we can see the traceback
    result = await execute_workflow(
        workflow=create_features_workflow,
        input={},
        namespace=TESTING_NAMESPACE,
        return_result=True,
        executor=executor,
    )


async def test_train_pipeline(train_pipeline_workflow, executor):
    result = await execute_workflow(
        workflow=train_pipeline_workflow,
        input={},
        namespace=TESTING_NAMESPACE,
        return_result=True,
        executor=executor,
    )


async def test_predict_pipeline(predict_pipeline_workflow, executor):
    result = await execute_workflow(
        workflow=predict_pipeline_workflow,
        input={},
        namespace=TESTING_NAMESPACE,
        return_result=True,
        executor=executor,
    )
