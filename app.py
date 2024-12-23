import os
from src.generated_assets import generate_assets
from src.lambda_handler import s3_handler
from dotenv import load_dotenv

load_dotenv()

artifacts_bucket = os.environ.get('ARTIFACTS_BUCKET')
performance_metrics_url = os.environ.get("PERFORMANCE_METRICS_URL")
performance_metrics_url_secret = os.environ.get("PERFORMANCE_METRICS_URL_SECRET")


def handler(event, context):
    """
        Lambda function handler.
    """
    (folder_name, dataset) = s3_handler(event)
    type_predict = 'landing_page_id'
    # type_predict = 'offer_id'
    status = generate_assets(artifacts_bucket, folder_name, dataset, type_predict, performance_metrics_url,
                             performance_metrics_url_secret)

    response = {
        'body': 'success' if status else 'failure',
        'statusCode': 200 if status else 500
    }
    return response
