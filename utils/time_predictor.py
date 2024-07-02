import time
import numpy as np
from datetime import datetime


class TimePredictor:
    def __init__(self, epochs, history=30):
        self.history = history
        self.rest_epochs = epochs
        self.duration_list = []
        self.prev_time = time.time()
        self.average_duration = 0.

    def update(self):
        temp_time = time.time()
        duration = temp_time - self.prev_time
        self.prev_time = temp_time
        self.duration_list.append(duration)
        self.rest_epochs -= 1
        if (len(self.duration_list) > self.history):
            self.duration_list = self.duration_list[-self.history:]
        self.average_duration = np.mean(self.duration_list)

    def get_predict(self):
        end_timestamp = self.prev_time + self.average_duration * self.rest_epochs
        return datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')
