import numpy
import torch
from threadpoolctl import threadpool_info

print(numpy.show_config())
print("------------------------")
print(torch.__config__.show())
print("------------------------")
print(threadpool_info())