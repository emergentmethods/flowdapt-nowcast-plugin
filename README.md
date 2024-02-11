# Flowdapt Nowcast Plugin

This Flowdapt plugin lets you run a now-casting use-case on a set of cities, models, and targets defined in the `weather_data.yaml`. Data is all sourced from the [OpenMeteo API](https://open-meteo.com/). This plugin was used to run the study found in "Balancing Computational Efficiency and Accuracy in Machine Learning-based Time-Series Forecasting: Insights from Live Experiments on Meteorological Nowcasting" submitted to NeurIPS 2023. The paper can be found [here](https://arxiv.org/abs/2309.15207).

## Usage

To run the following example commands, you must have Taskfile installed on your machine. To do so see the [Taskfile documentation](https://taskfile.dev/installation/).


Then clone this repository and run the following commands:

```bash
git clone https://gitlab.com/emergentmethods/flowdapt-nowcast-plugin.git
cd flowdapt-nowcast-plugin
task build
task run
```

This will build the plugin's docker image, and start it via docker compose. The flowdapt server will be available at `http://localhost:8080`. Next, in a new terminal window (with `flowctl` installed), run the following commands to ensure the workflows and configurations are applied:

```bash
flowctl apply -p workflows/
flowctl apply -p configs/
```

Then you can run the workflows:

```bash
flowctl run create_features
flowctl run train
flowctl run predict
```

However, if you want to run the plugin indefinitely to collect hourly predictions, you can use the `python_driver.py` in a separate terminal window or session (Note this script requires the python package `flowdapt_sdk` to be installed, nothing should be done if `flowctl` is installed):

```bash
python3 flowdapt_nowcast_plugin/python_driver.py
```

This will call the plugin's workflows via the Flowdapt Rest API every hour.

To stop the plugin, run the following command:

```bash
task stop
```

## Testing

To run the tests, run the following command:

```bash
task unit-tests
```
