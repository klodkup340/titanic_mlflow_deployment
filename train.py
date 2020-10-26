import os
import warnings
import sys

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import tempfile
import pickle
import shutil

import mlflow
import mlflow.sklearn
import mlflow.tracking

def log_mlflow(model):
  with mlflow.start_run() as run:

    # Logging params and metrics
    mlflow.set_tag('model_name', name)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('test_size', test_size)
    mlflow.log_param('CV_n_folds', CV_n_folds)
    mlflow.log_param('Train size', train_X.shape)
    mlflow.log_param('Columns', str(train_X.columns.values.tolist()))
    mlflow.log_metric('train_score', train_score)
    mlflow.log_metric('accuracy_score', accuracy)
    mlflow.log_metric('cv_mean_score', cv_mean_score)
    mlflow.log_metric('f1_score', f1_score)
    
    # Logging plotting picture to artifacts
    temp_file_name = get_temporary_directory_path("confusion_matrix-", ".png")
    temp_name = temp_file_name.name
    
    cm = confusion_matrix(pred, test_Y)
    cm_plot = sns.heatmap(cm, annot=True)
    fig = cm_plot.get_figure()

    try:
        fig.savefig(temp_name)
        mlflow.log_artifact(temp_name, "confusion_matrix_plots")
    finally:
        temp_file_name.close()  # Delete the temp file

    # Logging training dataset to artifacts
    tempdir = tempfile.mkdtemp()
    with open (os.path.join(tempdir, 'data.pkl'), 'wb') as t:
      pickle.dump({'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y}, t)
    
    mlflow.log_artifact(os.path.join(tempdir, 'data.pkl'), 'training_dataset')
    shutil.rmtree(tempdir)

    # Save model to artifacts
    mlflow.sklearn.log_model(model, name)
  mlflow.end_run()


def process_age(df, cut_points, label_names):
    df['Age'] = df['Age'].fillna(-0.5)
    df['Age_category'] = pd.cut(df['Age'], cut_points, labels=label_names)
    return df

def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

import tempfile

def get_temporary_directory_path(prefix, suffix):
  """
  Get a temporary directory and files for artifacts
  :param prefix: name of the file
  :param suffix: .csv, .txt, .png etc
  :return: object to tempfile.
  """

  temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix)
  return temp


if __name__ == "__main__":

    #Use sqlite:///mlruns.db as the local store for tracking and registry
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    data_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_train.csv")
    data_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_test.csv")

    data_train = pd.read_csv(data_train_path)
    data_test = pd.read_csv(data_test_path)

    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

    data_train = process_age(data_train, cut_points, label_names)
    data_test = process_age(data_test, cut_points, label_names)

    data_train = create_dummies(data_train, 'Pclass')
    data_test = create_dummies(data_test, 'Pclass')

    data_train = create_dummies(data_train, 'Age_category')
    data_test = create_dummies(data_test, 'Age_category')

    data_train = create_dummies(data_train, 'Sex')
    data_test = create_dummies(data_test, 'Sex')

    columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Sex_female', 'Age_category_Missing', 'Age_category_Infant',       'Age_category_Child', 'Age_category_Teenager', 'Age_category_Young Adult', 'Age_category_Adult', 'Age_category_Senior']

    all_X = data_train[columns]
    all_Y = data_train['Survived']

    ###########################
    test_size = 0.3
    CV_n_folds = 5
    ###########################

    train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=test_size, random_state=0)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    ## random forest algorithm
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)

    train_score = model.score(train_X, train_Y)
    accuracy = accuracy_score(test_Y, pred)
    cv_mean_score = np.mean(cross_val_score(model, all_X, all_Y, cv=CV_n_folds))
    f1_score = f1_score(test_Y, pred)

    name = 'RandomForestClassifier'


    log_mlflow(model)
    #print(sns.__version__)
    print('Tracking results are sent to Local Filesystem: ', mlflow.get_tracking_uri())
   





