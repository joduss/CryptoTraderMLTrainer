import datetime as dt


def dateparse (time_in_secs):
    return dt.datetime.utcfromtimestamp(float(time_in_secs))

def timestamp(time: dt.datetime) -> float:
    return time.timestamp()