"""Fixed formatting prompts, selected only on calibration examples."""

import re

SYSTEM = (
    "Follow the user's requested output format exactly. Return only the requested "
    "answer or value. Do not add explanations, equations, labels, quotation marks, "
    "or code fences. Do not translate words that the user asks you to copy."
)

EXAMPLES = [
    ("计算 12 + 7，只输出整数答案。", "19"),
    (
        "Sort these integers ascending. Output only comma-separated numbers: 9,2,7,4",
        "2,4,7,9",
    ),
    (
        '提取下面 JSON 的 code 值，只输出该值，不加引号：{"code": "R10234", "quantity": 3}',
        "R10234",
    ),
    (
        "Output word number 3 in this list (counting from 1). Output only that word: apple, river, stone, cloud",
        "stone",
    ),
]


TASK_EXAMPLES = {
    0: [
        ("计算 31 + 8，只输出整数答案。", "39"),
        ("What is 24 + 17? Output only the integer.", "41"),
        ("计算 63 + 9，只输出整数答案。", "72"),
        ("What is 46 + 13? Output only the integer.", "59"),
    ],
    1: [
        ("将这些整数从小到大排序，只输出逗号分隔的数字：8,31,4,20", "4,8,20,31"),
        (
            "Sort these integers ascending. Output only comma-separated numbers: 75,9,34,18",
            "9,18,34,75",
        ),
        ("将这些整数从小到大排序，只输出逗号分隔的数字：12,27,33,48", "12,27,33,48"),
        (
            "Sort these integers ascending. Output only comma-separated numbers: 93,80,26,7",
            "7,26,80,93",
        ),
    ],
    2: [
        (
            '提取下面 JSON 的 code 值，只输出该值，不加引号：{"code": "R54123", "quantity": 13, "price": 6}',
            "R54123",
        ),
        (
            'Extract the code value from this JSON. Output only its value without quotes: {"code": "R67234", "quantity": 25, "price": 14}',
            "R67234",
        ),
        (
            '提取下面 JSON 的 code 值，只输出该值，不加引号：{"code": "R89321", "quantity": 37, "price": 8}',
            "R89321",
        ),
        (
            'Extract the code value from this JSON. Output only its value without quotes: {"code": "R21546", "quantity": 40, "price": 3}',
            "R21546",
        ),
    ],
    3: [
        ("输出以下列表中第 1 个单词，只输出单词：apple, stone, river, cloud", "apple"),
        (
            "Output word number 4 in this list (counting from 1). Output only that word: light, house, quiet, green",
            "green",
        ),
        ("输出以下列表中第 3 个单词，只输出单词：river, apple, stone, cloud", "stone"),
        (
            "Output word number 2 in this list (counting from 1). Output only that word: house, quiet, green, light",
            "quiet",
        ),
    ],
}


def numbered(prompt, kind):
    if kind != 3:
        return prompt
    words = [word.strip() for word in re.split(r"[:：]", prompt)[-1].split(",")]
    if len(words) != 4 or any(not word.isalpha() for word in words):
        raise ValueError("The word-selection interface expects exactly four words")
    positions = "; ".join(f"{i + 1} = {word}" for i, word in enumerate(words))
    return (
        prompt
        + "\nThe supplied list with explicit positions: "
        + positions
        + ". Return only the word at the requested position."
    )


def messages(prompt, style="native", kind=None):
    if style not in {"native", "system", "fewshot", "taskfewshot", "tasknumbered"}:
        raise ValueError("Unknown frozen prompt style")
    result = [] if style == "native" else [{"role": "system", "content": SYSTEM}]
    if style in {"fewshot", "taskfewshot", "tasknumbered"}:
        examples = TASK_EXAMPLES[kind] if style in {"taskfewshot", "tasknumbered"} else EXAMPLES
        for question, answer in examples:
            if style == "tasknumbered":
                question = numbered(question, kind)
            result.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
    return result + [
        {
            "role": "user",
            "content": numbered(prompt, kind) if style == "tasknumbered" else prompt,
        }
    ]
