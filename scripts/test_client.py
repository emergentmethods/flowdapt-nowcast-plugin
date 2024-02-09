from __future__ import print_function
import time
import flowdapt_sdk
import asyncio
import cloudpickle
import pandas as pd
from pathlib import Path


async def main(historic_predictions):
    # Defining the host is optional and defaults to http://127.0.0.1:8080/api/latest
    # See configuration.py for a list of all supported configuration parameters.
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
                print("Running workflows...")
                print("Creating features...")
                api_response = await api_instance.run_workflow("openmeteo_create_features")
                print(f"create_features response: {api_response}")
                print("Training models...")
                api_response = await api_instance.run_workflow("openmeteo_train", _request_timeout=2400)
                print(f"train response: {api_response}")
                print("Getting predictions...")
                api_response = await api_instance.run_workflow("openmeteo_predict")
                print(f"prediction response: {api_response}")
                # print("The response of WorkflowsApi->get_workflow:\n")
                # pprint(api_response)

                # parse the response and save it to disk
                historic_predictions = parse_response(historic_predictions, api_response)

                with open("/srv/flowdapt/historic_predictions.pkl", "wb") as fp:
                    cloudpickle.dump(historic_predictions, fp,
                                     protocol=cloudpickle.DEFAULT_PROTOCOL)

                first = False
                await asyncio.sleep(5)
            else:
                print(f"Current time {current_time}")
                print(f"Waiting until {current_time.tm_min} is 1")
                await asyncio.sleep(20)
                # os.system("clear")

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

        # TODO shift ground_truth by respective data points 1, 4, 12

        if row["city"] not in historic_predictions:
            historic_predictions[row["city"]] = df
        else:
            historic_predictions[row["city"]] = pd.concat(
                [historic_predictions[row["city"]], df], ignore_index=True, axis=0)

    return historic_predictions


if __name__ == '__main__':
    id = "full_uscities"
    historic_predictions_path = Path(f"/srv/flowdapt/historic_predictions_{id}.pkl")

    if historic_predictions_path.exists():
        with historic_predictions_path.open("rb") as fp:
            historic_predictions = cloudpickle.load(fp)
    else:
        historic_predictions = {}

    asyncio.run(main(historic_predictions))
