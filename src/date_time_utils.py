import datetime


def get_time_stamp():
    try:
        now = datetime.datetime.now()
        seconds = int(now.timestamp())
        microseconds = now.microsecond
        time_stamp = seconds * 1000 + microseconds // 1000
        return time_stamp
    except Exception as e:
        print(e)
        print('Error')
        raise e
