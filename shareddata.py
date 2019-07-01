import numpy as np
import multiprocessing as mp
import time

sh = mp.RawArray('i', int(1e8))
x = np.arange(1e8, dtype=np.int32)
sh_np = np.ctypeslib.as_array(sh)

start = time.time()
start_cpu = time.clock()
sh_np[:] = x
end = time.time()
end_cpu = time.clock()
print("all time:", end)
print("CPU time:", end_cpu)

# sh[:] = x
# CPU times: user 10.1 s, sys: 132 ms, total: 10.3 s
# Wall time: 10.2 s

# memoryview(sh).cast('B').cast('i')[:] = x
# CPU times: user 64 ms, sys: 132 ms, total: 196 ms
# Wall time: 196 ms

# sh_np[:] = x
# CPU times: user 92 ms, sys: 104 ms, total: 196 ms
# Wall time: 196 ms