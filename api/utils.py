import configparser
import transformers
import numpy as np
import random

transformers.logging.set_verbosity_error()

def set_seed(seed):
  """Set seed globally."""
  random.seed(seed)
  np.random.seed(seed)
  try:
    import torch
    torch.manual_seed(seed)
  except:
    pass
  # try:
  #   import tensorflow as tf
  #   tf.random.set_seed(seed)
  # except:
  #   pass


def parse_optional_int(config, section, option):
  value = config.get(section, option)
  return int(value) if value is not None else None


def parse_optional_float(config, section, option):
  value = config.get(section, option)
  return float(value) if value is not None else None


def parse_optional_bool(config, section, option):
  value = config.get(section, option)
  return value.lower() in ("yes", "true", "t", "1") if value is not None else None


def parse_optional_int_list(config, section, option):
  value = config.get(section, option)
  return list(map(int, value.replace(' ', '').split(','))) if value is not None else None


def parse_config(config_path):
  """Parse config into a dict."""

  # Read the config
  config = configparser.ConfigParser(allow_no_value=True)
  with open(config_path) as f:
    config.read_file(f)

  return dict(
    general_params=dict(
      device=parse_optional_int(config, 'general_params', 'device'),
      seed=parse_optional_int(config, 'general_params', 'seed')
    ),
    generation_pipeline_kwargs=dict(
      model=config.get('generation_pipeline_kwargs', 'model'),
      config=config.get('generation_pipeline_kwargs', 'config'),
      tokenizer=config.get('generation_pipeline_kwargs', 'tokenizer'),
      framework=config.get('generation_pipeline_kwargs', 'framework')
    ),
    generator_kwargs=dict(
      max_length=parse_optional_int(config, 'generator_kwargs', 'max_length'),
      min_length=parse_optional_int(config, 'generator_kwargs', 'min_length'),
      do_sample=parse_optional_bool(config, 'generator_kwargs', 'do_sample'),
      early_stopping=parse_optional_bool(config, 'generator_kwargs', 'early_stopping'),
      num_beams=parse_optional_int(config, 'generator_kwargs', 'num_beams'),
      num_beam_groups=parse_optional_int(config, 'generator_kwargs', 'num_beam_groups'),
      diversity_penalty=parse_optional_float(config, 'generator_kwargs', 'diversity_penalty'),
      temperature=parse_optional_float(config, 'generator_kwargs', 'temperature'),
      top_k=parse_optional_int(config, 'generator_kwargs', 'top_k'),
      top_p=parse_optional_float(config, 'generator_kwargs', 'top_p'),
      repetition_penalty=parse_optional_float(config, 'generator_kwargs', 'repetition_penalty'),
      length_penalty=parse_optional_float(config, 'generator_kwargs', 'length_penalty'),
      no_repeat_ngram_size=parse_optional_int(config, 'generator_kwargs', 'no_repeat_ngram_size'),
      pad_token_id=parse_optional_int(config, 'generator_kwargs', 'pad_token_id'),
      bos_token_id=parse_optional_int(config, 'generator_kwargs', 'bos_token_id'),
      eos_token_id=parse_optional_int(config, 'generator_kwargs', 'eos_token_id'),
      bad_words_ids=parse_optional_int_list(config, 'generator_kwargs', 'bad_words_ids'),
      num_return_sequences=parse_optional_int(config, 'generator_kwargs', 'num_return_sequences'),
      decoder_start_token_id=parse_optional_int(config, 'generator_kwargs', 'decoder_start_token_id'),
      use_cache=parse_optional_bool(config, 'generator_kwargs', 'use_cache'),
      clean_up_tokenization_spaces=parse_optional_bool(config, 'generator_kwargs', 'clean_up_tokenization_spaces')
    ),
    prior_ranker_weights=dict(
      human_vs_rand_weight=parse_optional_float(config, 'prior_ranker_weights', 'human_vs_rand_weight'),
      human_vs_machine_weight=parse_optional_float(config, 'prior_ranker_weights', 'human_vs_machine_weight')
    ),
    cond_ranker_weights=dict(
      updown_weight=parse_optional_float(config, 'cond_ranker_weights', 'updown_weight'),
      depth_weight=parse_optional_float(config, 'cond_ranker_weights', 'depth_weight'),
      width_weight=parse_optional_float(config, 'cond_ranker_weights', 'width_weight')
    )
  )


def load_pipeline(task, **kwargs):
  """Load a pipeline."""
  return transformers.pipeline(task, **kwargs)


def clean_text(txt):
  """Remove unnecessary spaces."""
  return ' '.join(txt.strip().split())


def generate_responses(prompt, pipeline, seed=None, **kwargs):
  """Generate responses using a text generation pipeline."""
  if seed is not None:
    set_seed(seed)

  outputs = pipeline(prompt, **kwargs)
  responses = list(map(lambda x: clean_text(x['generated_text'][len(prompt):]), outputs))
  return responses


def build_ranker_dict(**kwargs):
  """Build dictionary of ranker weights and pipelines."""
  kwargs = kwargs.copy()
  human_vs_rand_weight = kwargs.pop('human_vs_rand_weight', None)
  human_vs_machine_weight = kwargs.pop('human_vs_machine_weight', None)
  updown_weight = kwargs.pop('updown_weight', None)
  depth_weight = kwargs.pop('depth_weight', None)
  width_weight = kwargs.pop('width_weight', None)

  ranker_dict = dict()
  if human_vs_rand_weight is not None:
    ranker_dict['human_vs_rand'] = dict(
      pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-human-vs-rand', **kwargs),
      weight=human_vs_rand_weight,
      group='prior'
    )
  if human_vs_machine_weight is not None:
    ranker_dict['human_vs_machine'] = dict(
      pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-human-vs-machine', **kwargs),
      weight=human_vs_machine_weight,
      group='prior'
    )
  if updown_weight is not None:
    ranker_dict['updown'] = dict(
      pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-updown', **kwargs),
      weight=updown_weight,
      group='cond'
    )
  if depth_weight is not None:
    ranker_dict['depth'] = dict(
      pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-depth', **kwargs),
      weight=depth_weight,
      group='cond'
    )
  if width_weight is not None:
    ranker_dict['width'] = dict(
      pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-width', **kwargs),
      weight=width_weight,
      group='cond'
    )
  return ranker_dict


def generate_scores(prompt, responses, pipeline, **kwargs):
  """Generate scores using a text classification pipeline."""
  responses = [prompt + response for response in responses]

  outputs = pipeline(responses, **kwargs)
  return [output['score'] for output in outputs]


def pick_best_response(prompt, responses, ranker_dict):
  """Pick the best response according to the weighted average of scores."""
  if len(ranker_dict) == 0:
    return random.choice(responses)

  def _get_wa_group_scores(group_name):
    group_scores = 0
    group_weight_sum = 0
    for model_name, dct in ranker_dict.items():
      if dct['group'] == group_name:
        scores = np.array(generate_scores(
          prompt,
          responses,
          dct['pipeline']
        ))
        group_scores += scores * dct['weight']
        group_weight_sum += dct['weight']
    group_scores /= group_weight_sum
    return group_scores

  group_names = list(map(lambda x: x['group'], ranker_dict.values()))
  if 'prior' in group_names:
    prior_scores = _get_wa_group_scores('prior')
  else:
    prior_scores = 1
  if 'cond' in group_names:
    cond_scores = _get_wa_group_scores('cond')
  else:
    cond_scores = 1
  final_scores = prior_scores * cond_scores
  return responses[np.argmax(final_scores)]
