import os
from sys import argv

os.mkdir("data")







formatted_context = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f'<|im_start|>assistant\n{{"name": "'
    )