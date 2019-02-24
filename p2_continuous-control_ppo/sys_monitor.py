import os
import psutil


def memory_usage_in_megabytes():
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    return current_memory/(1024.0*1024.0)
