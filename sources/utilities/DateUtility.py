import datetime as dt


def dateparse (time_in_secs):
    return dt.datetime.fromtimestamp(float(time_in_secs))