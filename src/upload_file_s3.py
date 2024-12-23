import os
import logging

from botocore.exceptions import ClientError

from src.s3_client.client import s3_client


def upload_file(bucket, folder_name, file_name):
    key_name = os.path.basename(file_name)

    try:
        object_name = folder_name + '/' + key_name
        s3_client().upload_file(file_name, bucket, object_name)
        print('Success Upload S3: ', key_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True