import re

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
one_score_pattern_another_format = re.compile("\[\[rating:(\d+)\]\]")
one_score_pattern_another_format2 =re.compile("\[\[rating: (\d+)\]\]")

text = "This is a sample text with [[rating: 42]] and [[rating: 3.14]] ratings."

match = re.search(one_score_pattern_another_format2,text)

if match:
    print("Matched text:", match.group())
    print("Captured rating:", match.group(1))
    print("Start position:", match.start())
    print("End position:", match.end())
    print("Match span:", match.span())
else:
    print("No match found.")