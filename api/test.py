from kazuma import generate
import time

while True:
  message = input('=> ')
  start = time.perf_counter()
  response = generate(message)
  end = time.perf_counter()
  print(response)
  print(f'Compute time: {end - start:.3f}s')
