from fastapi import FastAPI
from kazuma import generate

app = FastAPI()

@app.get('/')
def root():
  return {'endpoints': ['/kazuma']}

@app.get('/kazuma')
def kazuma(message: str):
  return {'response': generate(message)}
