import numpy as np
import time
import threading


class DataStream:
    def __init__(self, buffer_size=1024, sample_rate=1/6):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.buffer = np.zeros(buffer_size)
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.processors = []
        self.callbacks = []
        self.running = False
        self.source = None
        self.lock = threading.Lock()
        self.stats = {'total_samples': 0, 'processed': 0, 'errors': 0}

    def set_source(self, source_func):
        self.source = source_func

    def add_processor(self, processor):
        self.processors.append(processor)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def push_sample(self, sample):
        with self.lock:
            self.buffer[self.write_pos % self.buffer_size] = sample
            self.write_pos += 1
            self.count = min(self.count + 1, self.buffer_size)
            self.stats['total_samples'] += 1

    def process_sample(self, sample):
        processed = sample
        for proc in self.processors:
            try:
                processed = proc(processed)
            except Exception:
                self.stats['errors'] += 1
                return sample

        for cb in self.callbacks:
            try:
                cb(processed)
            except Exception:
                self.stats['errors'] += 1

        self.stats['processed'] += 1
        return processed

    def get_window(self, size=None):
        if size is None:
            size = self.buffer_size
        with self.lock:
            if self.count < size:
                return self.buffer[:self.write_pos].copy()
            start_pos = self.write_pos - size
            start_idx = start_pos % self.buffer_size
            end_idx = self.write_pos % self.buffer_size
            if start_idx < end_idx:
                return self.buffer[start_idx:end_idx].copy()
            else:
                return np.concatenate((self.buffer[start_idx:], self.buffer[:end_idx])).copy()

    def get_stats(self):
        return dict(self.stats)

    def start(self, duration=None):
        self.running = True
        start_time = time.time()
        while self.running:
            if self.source:
                sample = self.source()
                self.push_sample(sample)
                window = self.get_window(min(256, self.count))
                self.process_sample(window)
            if duration and (time.time() - start_time) >= duration:
                break
            time.sleep(self.sample_rate)

    def stop(self):
        self.running = False
