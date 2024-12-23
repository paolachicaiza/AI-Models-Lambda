import io
import urllib.parse
from src.s3_client.client import s3_client
import pandas as pd


def s3_handler(event):
    """
        This function is responsible for reading the data from S3.
    """

    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
        folder_name = key[key.index("/")+1:key.index(".")]
        dataset = s3_client().get_object(Bucket=bucket, Key=key)
        data_dataset = dataset['Body'].read()
        response = pd.read_csv(io.BytesIO(data_dataset),
                               header=0, delimiter=",",
                               low_memory=False)

        print("Uploaded Dataset: ", key)
        return folder_name, response
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}.'.format(key, bucket))
        raise e
