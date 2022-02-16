from kazuma import generate
import time

while True:
  start = time.perf_counter()
  response = generate(input('=> '))
  end = time.perf_counter()
  print(response)
  print(f'Compute time: {end - start:.3f}s')
