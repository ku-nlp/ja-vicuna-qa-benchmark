import json

for filename in ["gpt-3.5-davinci","llm-jp-1.3b","llm-jp-1.3b-refined","llm-jp-13b","llm-jp-13b-refined","llm-jp-13b-sft-js","llm-jp-13b-sft-dolly-oasst","llm-jp-13b-sft-js-dolly-oasst","llm-jp-13b-lora-sft-js-run1","llm-jp-13b-lora-sft-dolly-oasst-run1","llm-jp-13b-lora-sft-js-dolly-oasst-run1"]:
    print(filename)
    with open(filename+".jsonl","r") as f:
        nlen = 0
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            nlen += len(tmp_dict["choices"][0]["turns"][0])

        avg_len = nlen / 80.0
        print(avg_len)