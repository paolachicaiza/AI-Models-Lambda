import boto3
import os

class _S3ClientService:
  def __init__(self):
    self.client = boto3.client('s3', region_name='us-east-1')
    
  def __call__(self, *args, **kwds):
    return self.client

s3_client = _S3ClientService()