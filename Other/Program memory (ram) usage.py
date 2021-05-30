# Track how much memory your program uses  

import tracemalloc
tracemalloc.start()

# Your program

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()