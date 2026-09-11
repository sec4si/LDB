"""The same frozen task interface is used for evaluation and delivery."""


def messages(task):
    examples = (
        [
            ("What is 24 + 17? Output only the integer.", "41"),
            ("What is 63 + 9? Output only the integer.", "72"),
        ]
        if task["kind"] == "arithmetic"
        else [
            ('Extract code; output only its value: {"code":"X12954","count":4}', "X12954"),
            ('Extract code; output only its value: {"code":"X53728","count":9}', "X53728"),
        ]
    )
    result = [
        {
            "role": "system",
            "content": "Return only the requested answer. No explanation or extra formatting.",
        }
    ]
    for question, answer in examples:
        result += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return result + [{"role": "user", "content": task["prompt"]}]
