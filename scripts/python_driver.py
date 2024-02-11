"""
An example script designed to interact with an active flowdapt server
via the flowdapt SDK.

The script automates the launching workflows once per hour.
- create_features
- train
- predict

It takes the response from the predict workflow, parses it and saves it to
a historic_predictions.pkl file for future post-processing.
"""
from __future__ import print_function
import time
import flowdapt_sdk
import asyncio
import cloudpickle
import pandas as pd
from pathlib import Path

SERVER_URL = "http://localhost:8080"
STUDY_IDENTIFIER = "full_uscities"
HISTORIC_PREDICTIONS_PATH = Path(f"/srv/flowdapt/historic_predictions_{STUDY_IDENTIFIER}.pkl")


async def main(historic_predictions):
    async with flowdapt_sdk.FlowdaptSDK(base_url=SERVER_URL, timeout=2400.0) as sdk:
        # Enter a context with an instance of the API Client
        first = True
        while True:
            current_time = time.localtime()
            if first or (current_time.tm_min == 1):
                print("Running workflows...")
                print("Creating features...")
                api_response = await sdk.workflows.run_workflow("openmeteo_create_features")
                print(f"create_features response: {api_response}")
                print("Training models...")
                api_response = await sdk.workflows.run_workflow("openmeteo_train")
                print(f"train response: {api_response}")
                print("Getting predictions...")
                api_response = await sdk.workflows.run_workflow("openmeteo_predict")
                print(f"prediction response: {api_response}")
                # print("The response of WorkflowsApi->get_workflow:\n")
                # pprint(api_response)

                # parse the response and save it to disk
                historic_predictions = parse_response(historic_predictions, api_response)

                HISTORIC_PREDICTIONS_PATH.write_bytes(
                    cloudpickle.dumps(historic_predictions, protocol=cloudpickle.DEFAULT_PROTOCOL)
                )

                first = False
                await asyncio.sleep(5)
            else:
                print(f"Current time {current_time}")
                print(f"Waiting until {current_time.tm_min} is 1")
                await asyncio.sleep(20)


def parse_response(historic_predictions: dict, api_response: dict):
    result = api_response["result"]["return_predictions"]
    for item in result:
        row = item[0]
        preds = item[1][0]
        labels = item[3]
        dates = [item[2]]
        ground_truth = item[4]
        truth_labs = item[5]

        labels.extend(truth_labs)
        labels.insert(0, "dates")

        # Convert preds to a Pandas dataframe
        df = pd.DataFrame(data=[dates + preds + ground_truth], columns=labels)

        name_str = row['city']
        target = row['target']

        if name_str not in historic_predictions:
            historic_predictions[name_str] = {}

        if target not in historic_predictions[name_str]:
            historic_predictions[name_str][target] = df
        elif historic_predictions[name_str][target]["dates"].iloc[-1] == df["dates"].iloc[-1]:
            print(f"Already have a prediction for {row['city']} on {df['dates'].iloc[-1]}, "
                  "skipping historic_predictions update")
            continue
        else:
            historic_predictions[name_str][target] = pd.concat(
                [historic_predictions[name_str][target], df], ignore_index=True, axis=0)

    return historic_predictions


if __name__ == '__main__':
    if HISTORIC_PREDICTIONS_PATH.exists():
        historic_predictions = cloudpickle.load(
            HISTORIC_PREDICTIONS_PATH.read_bytes()
        )
    else:
        historic_predictions = {}

    asyncio.run(main(historic_predictions))
