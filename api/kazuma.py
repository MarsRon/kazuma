from utils import parse_config, load_pipeline, build_ranker_dict, clean_text, generate_responses, pick_best_response

# Config model
config = parse_config('kazuma.cfg')

# Extract parameters
general_params = config.get('general_params', {})
device = general_params.get('device', -1)
seed = general_params.get('seed', None)

generation_pipeline_kwargs = config.get('generation_pipeline_kwargs', {})
generator_kwargs = config.get('generator_kwargs', {})

prior_ranker_weights = config.get('prior_ranker_weights', {})
cond_ranker_weights = config.get('cond_ranker_weights', {})

# Prepare the pipelines
generation_pipeline = load_pipeline('text-generation', device=device, **generation_pipeline_kwargs)
ranker_dict = build_ranker_dict(device=device, **prior_ranker_weights, **cond_ranker_weights)

def generate(user_message):
  """Generate AI response"""

  # Add EOS token
  prompt = clean_text(user_message) + generation_pipeline.tokenizer.eos_token

  # Generate bot messages
  bot_messages = generate_responses(
    prompt,
    generation_pipeline,
    seed=seed,
    **generator_kwargs
  )

  # Pick best message
  if len(bot_messages) == 1:
    bot_message = bot_messages[0]
  else:
    bot_message = pick_best_response(
      prompt,
      bot_messages,
      ranker_dict
    )

  return bot_message
