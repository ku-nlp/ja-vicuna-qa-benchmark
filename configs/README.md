# Configuration files

Each configuration file is a JSON file with the following structure:

```json5
// rinna--japanese-gpt-neox-3.6b-instruction-ppo.json
{
  // The ID of the model
  "model_id": "rinna--japanese-gpt-neox-3.6b-instruction-ppo",
  // The name of the model
  "model_name_or_path": "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
  // The name of the lora model (optional)
  "lora_model_name_or_path": null,
  // The name of the tokenizer (optional)
  "tokenizer_name_or_path": null,
  // The prompt template
  "prompt_template": "ユーザー: {instruction}<NL>システム: ",
  // The generation configuration (optional)
  // NOTE: `temperature` will be set to a default value for each task category if left empty
  "generation_config": {
    "do_sample": true,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
  },
  // The special token map (optional); this is used to replace special tokens in the output
  "special_token_map": {
    "<NL>": "\n"
  }
}
```
