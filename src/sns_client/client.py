import json
import boto3
import os 

class _SNSClientService:
  def __init__(self):
    self.TargetArn = os.environ.get('ARN_SNS_TOPIC')
    self.client = boto3.client('sns', region_name='us-west-2')
  
  def __call__(self, *args, **kwds):
    return self.client

  def send_notification(self, notification):
    return self.client.publish(
      TargetArn=self.TargetArn,
      Message = json.dumps({'default': json.dumps(notification)}),
      MessageStructure = 'json'
    )

sns_client = _SNSClientService()
