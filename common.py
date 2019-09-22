from datetime import datetime

import math
from dateutil.relativedelta import relativedelta


def format_seconds(seconds: float) -> str:
    seconds = math.fabs(seconds)
    if seconds > 3600:
        return f"{seconds / 3600} hours"
    elif seconds > 60:
        return f"{seconds / 60} minutes"
    else:
        return f"{seconds} seconds"


def format_time_lapse(start: float, end: float) -> str:
    attrs = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
    delta = relativedelta(datetime.fromtimestamp(end), datetime.fromtimestamp(start))
    spaces = ['%d %s' % (getattr(delta, attr), getattr(delta, attr) > 1 and attr or attr[:-1]) for attr in attrs if
              getattr(delta, attr)]
    return ", ".join(spaces)
