import os
import datetime


class Logger:
    def __init__(self, log_fold : str):
        if os.path.exists(log_fold) == False:
            os.makedirs(log_fold)

        self.log_fold = log_fold
    def format_time(self):
        time = datetime.datetime.now()
        
        hour = time.hour
        minute = time.minute
        second = time.second

        day = time.day
        month = time.month
        year = time.year

        return '{}_{}_{}'.format(day, month, year), \
            '{}:{}:{}'.format(hour, minute, second)

    def log(self, content : str, show : bool = False):
        day, time = self.format_time()
        day = '{}/{}.txt'.format(self.log_fold, day)

        content = '{} - {}'.format(time, content)
        if show:
            print(content)

        f = open(day, 'a', encoding = "utf-8")
        f.write('\n' + content)
        f.close()