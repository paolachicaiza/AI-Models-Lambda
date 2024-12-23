import json
import time
import requests

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

from src.create_encoder import create_encoder
from src.date_time_utils import get_time_stamp
from src.imputation_data import imputation_data
from src.metrics_reports import classes_name, classes_report
from src.upload_file_s3 import upload_file
from src.model_mlp import model_mlp
from src.sns_client.client import sns_client


def generate_assets(bucket, folder_name, dataset, type_predict, url, url_secret):
    """
        Generate artifacts for a given dataset.
    """
    try:

        data_columns, total_rows, x, y = imputation_data(dataset,type_predict)
        start = time.time()
        x_values = x
        y_values = y
        y_classes = y
        number_landing_page_id = len(y_classes.drop_duplicates())
        x = x.astype(str).to_numpy()
        y = y.to_numpy()
        encoded_x = None
        for i in range(0, x.shape[1]):
            label_encoder = LabelEncoder()
            feature = label_encoder.fit_transform(x[:, i])
            feature = feature.reshape(x.shape[0], 1)
            onehot_encoder = OrdinalEncoder()
            feature = onehot_encoder.fit_transform(feature)
            if encoded_x is None:
                encoded_x = feature
            else:
                encoded_x = np.concatenate((encoded_x, feature), axis=1)

        x_encode = pd.DataFrame(encoded_x, columns=data_columns)

        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(y)
        label_encoded_y = label_encoder.transform(y)
        dummy_y = np_utils.to_categorical(label_encoded_y)
        y_encode = pd.DataFrame(label_encoded_y, columns=[type_predict])

        df_encoded_x = pd.DataFrame(encoded_x)
        df_encoded_x.to_csv('/tmp/encode_data_training.csv')
        upload_file(bucket, folder_name, '/tmp/encode_data_training.csv')

        df_encoded_y = pd.DataFrame(dummy_y)
        df_encoded_y.to_csv('/tmp/encode_data_training_y.csv')
        upload_file(bucket, folder_name, '/tmp/encode_data_training_y.csv')

        map_encode_data_input = create_encoder(x_values, x_encode)
        map_encoder = json.dumps(map_encode_data_input, indent=3)
        with open("/tmp/map_encode_data_input.json", "w") as outfile:
            outfile.write(map_encoder)
        upload_file(bucket, folder_name, '/tmp/map_encode_data_input.json')

        map_encode_data_output = create_encoder(y_encode, y_values)
        map_encoder = json.dumps(map_encode_data_output, indent=3)
        with open("/tmp/map_encode_data_output.json", "w") as outfile:
            outfile.write(map_encoder)
        upload_file(bucket, folder_name, '/tmp/map_encode_data_output.json')

        x_train, x_test, y_train, y_test = train_test_split(encoded_x, dummy_y, test_size=0.30, random_state=123)
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model = model_mlp(x_train, number_landing_page_id)

        history = model.fit(
            x_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(x_test, y_test),
            shuffle=False,
            verbose=2  # Only enable for debugging purposes
        )

        model.evaluate(x_train, y_train, verbose=False)
        model.evaluate(x_test, y_test, verbose=False)

        model.save('/tmp/model_multiclass.h5')
        upload_file(bucket, folder_name, '/tmp/model_multiclass.h5')

        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        target_names = classes_name(map_encode_data_output, type_predict)
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        model_accuracy = report["accuracy"]
        report_performance_classes = classes_report(report, target_names)

        time_stamp = get_time_stamp()
        fin = time.time()
        time_training_generate_assets = fin - start

        data = {
            "documentID": folder_name,
            "created_on": time_stamp,
            "accuracy": model_accuracy,
            "type": type_predict,
            "performance": report_performance_classes,
            "dataset_info": {
                "records_count": total_rows,
                "balanced": "true",
            },
            "duration_time": time_training_generate_assets
        }

        model_parameters = json.dumps(data, indent=3)
        with open("/tmp/model_parameters.json", "w") as outfile:
            outfile.write(model_parameters)
        upload_file(bucket, folder_name, '/tmp/model_parameters.json')

        data_columns = {
            "data_entries": data_columns,
            "type_predict": type_predict
        }

        data_columns_entries = json.dumps(data_columns, indent=3)
        with open("/tmp/data_columns_entries.json", "w") as outfile:
            outfile.write(data_columns_entries)
        upload_file(bucket, folder_name, '/tmp/data_columns_entries.json')

        header = {'Content-Type': 'application/json', 'Authorization': f"{url_secret}"}

        data_post = requests.post(url, json=data, headers=header)

        print(data_post.text)

        #####################################################
        #                                                   #
        #   send notification to refresh models in redis    #
        #                                                   #
        #####################################################

        sns_client.send_notification({
            "bucket": {
                "name": bucket,
                "folder": folder_name
            }
        }
        )

        #####################################################
        #                                                   #
        #   TO-DO: Upload files to Mongodb instead of S3    #
        #                                                   #
        #####################################################

        return True

    except Exception as error:
        print(error)
        return False
