import os
import json
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from fbprophet import Prophet
from fbprophet.serialize import model_to_json

from azureml.core import Workspace, Dataset
from azureml.core.run import Run

train_dataset_name = 'sales_train1'
test_dataset_name = 'sales_test1'
holiday_dataset_name = 'sales_holidays'

output_folder = 'outputs'


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # there is a problem in test dataset where Sales = 0
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    # run instance from the runtime experiment
    run = Run.get_context()

    # run remotely
    ws = run.experiment.workspace

    # run locally
    #ws = Workspace.from_config(path='../config.json')

    # get train, test, holidays datasets
    train_df = Dataset.get_by_name(
        ws, name=train_dataset_name).to_pandas_dataframe()[['Date', 'Sales']]
    train_df['Date'] = pd.DatetimeIndex(train_df['Date'])
    print('train_df', train_df)

    test_df = Dataset.get_by_name(
        ws, name=test_dataset_name).to_pandas_dataframe()[['Date', 'Sales']]
    test_df['Date'] = pd.DatetimeIndex(test_df['Date'])
    print('test_df', test_df)

    holiday_df = Dataset.get_by_name(
        ws, name=holiday_dataset_name).to_pandas_dataframe()
    holiday_df['ds'] = pd.DatetimeIndex(holiday_df['ds'])

    # Fix this issue: raise KeyError(f"{labels} not found in axis") - KeyError: '[] not found in axis'
    # https://www.jianshu.com/p/cd2d0cb63008
    # https://github.com/facebook/prophet/issues/821#issuecomment-461996281
    holiday_df = holiday_df.reset_index()
    print('holiday_df', holiday_df)

    # from the Prophet documentation every variables should have specific names
    train_df = train_df.rename(columns={
        'Date': 'ds',
        'Sales': 'y'
    })
    print('train_df', train_df)

    test_df = test_df.rename(columns={
        'Date': 'ds',
        'Sales': 'y'
    })
    print('test_df', test_df)

    # add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--seasonality_mode', type=str, default='additive',
                        help='Model seasonality mode')

    parser.add_argument('--changepoint_prior_scale', type=float, default=0.05,
                        help='How flexible the changepoints are allowed to be')

    parser.add_argument('--holidays_prior_scale', type=float, default=10.0,
                        help='Used to smoothning the effect of holidays')

    parser.add_argument('--n_changepoints', type=int, default=25,
                        help='Number of change happen in the data')

    args = parser.parse_args()
    print('args', args)

    model = Prophet(
        changepoint_prior_scale=args.changepoint_prior_scale,
        holidays_prior_scale=args.holidays_prior_scale,
        n_changepoints=args.n_changepoints,
        seasonality_mode=args.seasonality_mode,
        weekly_seasonality=True,
        daily_seasonality=True,
        yearly_seasonality=True,
        holidays=holiday_df,
        interval_width=0.95
    )

    model.fit(train_df)

    # make future dates the same as test dataset
    test_forecast = model.make_future_dataframe(
        periods=(test_df['ds'].max() - test_df['ds'].min()).days + 1,
        freq='D',
        include_history=False
    )
    print('test_forecast', test_forecast)

    # predict test
    test_predictions = model.predict(test_forecast)
    print('test_predictions', test_predictions)

    test_predictions = test_predictions[['ds', 'yhat']]
    print('test_predictions', test_predictions)

    # mean_absolute_percentage_error(test_df['y'], abs(test_predictions['yhat']))
    score_value = r2_score(test_df['y'], test_predictions['yhat'])
    print('score_value', score_value)

    run.log('score_value', np.float(score_value))

    # files saved to 'outputs' folder will be automatically uploaded to run history
    os.makedirs(output_folder, exist_ok=True)

    with open(f'{output_folder}/model.json', 'w') as fout:
        json.dump(model_to_json(model), fout)  # Save model


# standalone run: python train.py --seasonality_mode=additive --changepoint_prior_scale=0.05 --holidays_prior_scale=10.0 --n_changepoints=25
if __name__ == '__main__':
    main()
