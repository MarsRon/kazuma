from fastapi import FastAPI
from kazuma import generate
import time
import uvicorn
from os import environ

app = FastAPI()
port = int(environ.get('PORT', 3000))

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

uvicorn.run(app, host='0.0.0.0', port=port)
