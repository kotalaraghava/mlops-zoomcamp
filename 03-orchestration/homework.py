from pkgutil import get_data
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
from prefect import flow, task
import prefect
import pickle

# logger = prefect.get_run_logger(context)

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@task
def get_path(date):
    train_file = "./data/fhv_tripdata_{}-{}.parquet"
    val_path = "./data/fhv_tripdata_{}-{}.parquet"
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    if date is None:
        date = datetime.datetime.now()
    
    dt = (date - datetime.timedelta(days=30))
    if dt.month > 9:
        month1 = f"{dt.month}"
    else:
        month1 = f"0{dt.month}"
    if dt.month -1 > 9:
        month2 = f"{dt.month-1}"
    else:
        month2 = f"0{dt.month-1}"
    train_file = train_file.format(dt.year, month2)
    val_path = val_path.format(dt.year, month1)
    return train_file, val_path

@flow
def main_homework(date=None):

    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path = get_path(date).result()
    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    lr_file_name = f"./artifacts/model-{date}.bin"
    dv_file_name = f"./artifacts/dv-{date}.b"
    with open(lr_file_name, "wb") as w:
        pickle.dump(lr, w)
    with open(dv_file_name, "wb") as w:
        pickle.dump(dv, w)

    run_model(df_val_processed, categorical, dv, lr)

# main(date="2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
# from datetime impot

DeploymentSpec(
    flow=main_homework,
    name="model_training_homework",
    schedule = CronSchedule(cron="0 9 15 * *",timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml', 'working']
)

#9 Am every 15th of month.