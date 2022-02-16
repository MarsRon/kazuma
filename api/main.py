from fastapi import FastAPI
from kazuma import generate
import time
import uvicorn

app = FastAPI()

@app.get('/')
def root():
  return {'endpoints': ['/kazuma']}

@app.get('/kazuma')
def kazuma(message: str):
  start = time.perf_counter()
  response = generate(message)
  end = time.perf_counter()
  return {
    'response': response,
    'compute_time': end - start
  }

uvicorn.run(app)
