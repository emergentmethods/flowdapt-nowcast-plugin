from __future__ import print_function
import time
import flowdapt_sdk
import asyncio
import cloudpickle
import pandas as pd
from pathlib import Path

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


async def main(historic_predictions):
    # Assuming your flowdapt instance is running locally
    configuration = flowdapt_sdk.Configuration(
        host="http://127.0.0.1:8080/api/latest"
    )
    # Enter a context with an instance of the API client
    async with flowdapt_sdk.ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = flowdapt_sdk.WorkflowsApi(api_client)
        first = True
        while True:
            current_time = time.localtime()
            if first or (current_time.tm_min == 1):
                try:
                    await asyncio.sleep(3)
                    print("Running workflows...")
                    print("Creating features...")
                    api_response = await api_instance.run_workflow("openmeteo_create_features")
                    print(f"create_features response: {api_response}")
                    print("Training models...")
                    api_response = await api_instance.run_workflow("openmeteo_train", _request_timeout=3600)
                    print(f"train response: {api_response}")
                    print("Getting predictions...")
                    api_response = await api_instance.run_workflow("openmeteo_predict")
                    print(f"prediction response: {api_response}")

                    # parse the response and save it to disk
                    historic_predictions = parse_response(historic_predictions, api_response)

                    with open(f"user_data/historic_predictions_{id}.pkl", "wb") as fp:
                        cloudpickle.dump(historic_predictions, fp,
                                         protocol=cloudpickle.DEFAULT_PROTOCOL)

                    first = False
                    await asyncio.sleep(5)
                except Exception as e:
                    print(f"Exception when calling WorkflowsApi->run_workflow {e}")
            else:
                print(f"Current time {current_time}")
                print(f"Waiting until {current_time.tm_min} is 1")
                await asyncio.sleep(20)


def parse_response(historic_predictions: dict, api_response: dict):
    """
    Parses the response from the server so that we can save real-time
    predictions in a dataframe for post-processing.
    """
    result = api_response["result"]
    for key in result:
        item = key['preds']
        row = item[0]
        preds = item[1][0]
        labels = item[3]
        dates = [item[2]]
        ground_truth = item[4]
        truth_labs = item[5]

        labels.extend(truth_labs)
        labels.insert(0, "dates")

        # Convert preds payload to a Pandas dataframe
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
    id = "full_cities11"
    historic_predictions_path = Path(f"user_data/historic_predictions_{id}.pkl")

    if historic_predictions_path.exists():
        with historic_predictions_path.open("rb") as fp:
            historic_predictions = cloudpickle.load(fp)
    else:
        historic_predictions = {}

    asyncio.run(main(historic_predictions))
