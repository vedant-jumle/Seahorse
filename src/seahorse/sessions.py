"""Chat-sequence builders for the two-session experiment."""


def chat_ids(tok, user_text):
    """Token ids for a single user turn followed by the assistant-start tokens."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]


def text_ids(tok, text):
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]


def common_prefix_len(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def common_suffix_len(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def write_pair(tok, experience, followup):
    """Session-1 sequences with and without the experience.

    Returns (with_ids, without_ids, n): the last n tokens of both sequences are the
    shared suffix after the experience (follow-up, end of user turn, assistant
    header). These are the tokens the memory is written from. Using the longest
    common suffix avoids tokenisation-boundary mismatches.
    """
    with_ids = chat_ids(tok, f"{experience} {followup}")
    without_ids = chat_ids(tok, followup)
    n = common_suffix_len(with_ids, without_ids)
    assert n > 0, "no shared suffix between with/without sequences"
    return with_ids, without_ids, n


def ceiling_ids(tok, experience, probe):
    """Session-2 probe with the experience in context (the upper bound)."""
    return chat_ids(tok, f"{experience} {probe}")
