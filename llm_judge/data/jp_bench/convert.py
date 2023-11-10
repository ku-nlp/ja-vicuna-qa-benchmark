import json


import sys

dict_list = []
infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, "r") as f:
    for line in f.read().splitlines():
    #data = json.load(f)
    #for tmp_dict in data:
        tmp_dict = json.loads(line)
        tmp_dict["question_id"] = int(tmp_dict["question_id"])
        tmp_dict["turns"] = [tmp_dict["text"]]
        del tmp_dict["text"]


        with open(outfile, "a") as f:
            json.dump(tmp_dict, f, ensure_ascii=False)
            f.write("\n")
