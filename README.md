# Flowdapt Nowcast Plugin

The plug-in lets you run a now-casting use-case on a set of cities, models, and targets defined in the `weather_data.yaml`. Data is all sourced from the [OpenMeteo API](https://open-meteo.com/). This plug-in was used to run the study found in "Balancing Computational Efficiency and Accuracy in Machine Learning-based Time-Series Forecasting: Insights from Live Experiments on Meteorological Nowcasting" submitted to NeurIPS 2023. The paper can be found [here](https://arxiv.org/abs/2309.15207).

## Usage

After the plug-in is installed (see below), you can run the following commands (assuming `flowdapt run` was launched in a separate terminal to start the server):

First make sure you have applied the workflows and configs:

```bash
flowctl apply -p flowdapt_nowcast_plugin/workflows
flowctl apply -p flowdapt_nowcast_plugin/configs
```

Then you can run the workflows:

```bash
flowctl run create_features
flowctl run train
flowctl run predict
```

However, if you want to run the plug-in indefinitely to collect hourly predictions, you can use the `python_driver.py`:

```bash
python3 flowdapt_openmeteo_plugin/python_driver.py
```

## Installation

For use, the user is recommended to install the plugin via the commands in flowdapt:

```bash
$ pip install flowdapt-nowcast-plugin
```

For development, run the following commands:
```bash
$ git clone git@gitlab.com/emergentmethods/flowdapt-nowcast-plugin.git
$ cd flowdapt-nowcast-plugin
$ python3 -m venv .venv
$ source .venv/bin/activate
  # Make sure poetry is installed
$ curl -sSL https://install.python-poetry.org | python3 -
$ poetry install
$ pre-commit install
```

To test the stages and workflows, you have 2 options:

1. Run the command `pytest`, this will run the test suite which includes `test_stages.py` where the actual stage functions are tested in dummy workflows. The create_features, train, and predict workflows all run together in a session to  ensure the stage functions are working.
