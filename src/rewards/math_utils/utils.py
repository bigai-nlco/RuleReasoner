"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""

import re
import math
from typing import List, Optional

import sympy
from thefuzz.fuzz import ratio
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from lingua import Language, LanguageDetectorBuilder


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_generation_length(given_answer: str) -> float:
    if len(given_answer) < 64:
        return -0.5
    elif len(given_answer) < 128:
        return 0.1
    elif len(given_answer) < 256:
        return 0.2
    elif len(given_answer) < 512:
        return 0.3
    return 0.0


def grade_language_monotony(given_answer: str, language: str = "zh") -> bool:
    if language == "zh":
        target_language = "CHINESE"
    elif language == "en":
        target_language = "ENGLISH"
    else:
        raise ValueError(f"Language {language} must be specified correctly.")

    detector = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )
    confidence_list = detector.compute_language_confidence_values(given_answer)
    lang2conf = {
        confidence.language.name: confidence.value for confidence in confidence_list
    }
    if lang2conf[target_language] < 0.8:
        return False

    return True


def grade_language_repetition(
    given_answer: str,
    language: str = "zh",
    ngram: int = 2,
    tau: float = 1.0,
    steepness: float = 4.0,
) -> float:
    """
    Calculate a smoothed diversity reward based on distinct-n score for the given text,
    with temperature scaling to control the influence of the reward.

    Args:
        given_answer (str): The text to evaluate
        language (str): Language code, default "zh" for Chinese
        ngram (int): Size of n-grams to use, default 2
        tau (float): Temperature parameter in range [0, 1] to control reward scaling, default 1.0
                    - tau = 0: No diversity reward (always returns 0)
                    - tau = 1: Full diversity reward (returns value in [-1, 0])
                    - 0 < tau < 1: Scaled diversity reward

    Returns:
        float: A scaled reward value between -1 and 0, where values closer to 0 indicate higher diversity
    """
    # Ensure tau is in valid range
    tau = max(0.0, min(1.0, tau))

    # If tau is 0, diversity doesn't matter, return 0 reward
    if tau == 0:
        return 0.0

    # Check if input is empty
    if not given_answer or len(given_answer.strip()) == 0:
        return -1.0 * tau  # Minimum reward for empty text, scaled by tau

    # Chinese tokenization
    if language == "zh":
        try:
            import jieba

            tokens = list(jieba.cut(given_answer))
        except ImportError:
            # Fallback: simple character-based tokenization for Chinese
            tokens = list(given_answer)
    else:
        # For other languages, split by whitespace (simple approach)
        tokens = given_answer.split()

    # Generate n-grams
    ngrams = []
    for i in range(len(tokens) - ngram + 1):
        ngrams.append(tuple(tokens[i : i + ngram]))

    # Calculate distinct-n score
    if not ngrams:
        return -1.0 * tau  # Minimum reward if no n-grams could be formed, scaled by tau

    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))

    # Distinct-n score: ratio of unique n-grams to total n-grams
    distinct_n = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

    # Smoothing function to map distinct-n (range 0-1) to reward (range -1 to 0)
    # Using a sigmoid-like function that gives more reward as diversity increases
    # and approaches 0 (max reward) as distinct_n approaches 1

    # Parameters to tune the smoothing function
    steepness = steepness  # Controls how steep the reward curve is
    midpoint = 0.5  # The distinct-n value that gives a reward of -0.5

    # Sigmoid-like function mapped to [-1, 0]
    raw_reward = -1 + 1 / (1 + math.exp(-(math.e**steepness) * (distinct_n - midpoint)))

    # Apply temperature scaling - scales the reward by tau
    scaled_reward = raw_reward * tau

    # Ensure the reward stays within [-1, 0]
    scaled_reward = max(-1, min(0, scaled_reward))

    return scaled_reward


def parse_generation(
    preds: List[str],
) -> List[str]:
    """parse the generated texts to extract the final answer.

    Args:
        preds (List[str]): generated texts

    Returns:
        List[str]: parsed texts
    """
    regex_list = [
        r"<answer>\[刑期\](.*?)</answer>",
        r"<answer>\[金额\](.*?)</answer>",
        r"<answer>\n\[刑期\](.*?)\n</answer>",
        r"<answer>\n\[金额\](.*?)\n</answer>",
        r"<answer>\\n\[刑期\](.*?)\\n</answer>",
        r"<answer>\\n\[金额\](.*?)\\n</answer>",
    ]
    parsed_answers = []
    for pred in preds:
        parsed_answer = ""
        for regex in regex_list:
            match = re.search(regex, pred)
            if match and match.group(1):
                parsed_answer = match.group(1)
                break
        parsed_answers.append(parsed_answer.strip())

    return parsed_answers


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def grade_law_solution_by_outcome(
    pred: str,
    label: str,
    enable_soft_match: bool = True,
    enable_fuzzy_match: bool = False,
) -> bool:
    """compute the accuracy of the model predictions.

    Args:
        pred (str): predicted text
        label (str): ground-truth text
        enable_soft_match (bool, optional): enable soft match. Defaults to False.
        enable_fuzzy_match (bool, optional): enable fuzzy match. Defaults to True.

    Returns:
        float: accuracy
    """

    def is_digit_equal(pred: str, label: str) -> bool:
        pred_digit = (
            (
                pred.replace("年", "")
                .replace("月", "")
                .replace("日", "")
                .replace("亿", "")
            )
            .replace("千万", "")
            .replace("百万", "")
            .replace("十万", "")
            .replace("万", "")
            .replace("千", "")
            .replace("百", "")
        )
        label_digit = (
            (
                label.replace("年", "")
                .replace("月", "")
                .replace("日", "")
                .replace("亿", "")
            )
            .replace("千万", "")
            .replace("百万", "")
            .replace("十万", "")
            .replace("万", "")
            .replace("千", "")
            .replace("百", "")
        )
        return pred_digit == label_digit

    def is_soft_match(
        pred: str, label: str, sample_type: str, float_range: int = 3
    ) -> bool:
        if sample_type == "量刑":
            both_month_unit = "月" in pred and "月" in label
            pred = pred.replace("年", "").replace("月", "").replace("日", "")
            label = label.replace("年", "").replace("月", "").replace("日", "")
            if pred.isdigit() and label.isdigit():
                label_int = int(label)
                pred_int = int(pred)
                label_upper = min(label_int + float_range, 12)
                label_lower = max(label_int - float_range, 0)
                # NOTE: no need to consider the strings like 3年10个月 vs. 4年1个月 in LawGPT
                if label_lower <= pred_int <= label_upper:
                    return True
                elif both_month_unit and label_int == pred_int:
                    return True
                else:
                    return False
        return False

    if not label or not pred:
        return False

    if label == "" or pred == "":
        return False

    # exact match trial
    sample_type = "量刑" if "[刑期]" in label else "罚金"

    parsed_pred = parse_generation([pred])[0]
    parsed_label = parse_generation([label])[0]
    # print(f"parsed_pred: {parsed_pred}, parsed_label: {parsed_label}")

    if parsed_pred == "" or parsed_label == "":
        # empty prediction
        return False
    elif parsed_pred == parsed_label or is_digit_equal(parsed_pred, parsed_label):
        # exact match trial
        return True
    elif enable_fuzzy_match:
        # fuzzy match trial (if enabled)
        if ratio(parsed_pred, parsed_label) > 90:
            return True
    elif enable_soft_match:
        # soft match trial (if enabled)
        if is_soft_match(parsed_pred, parsed_label, sample_type):
            return True

    return False


def grade_law_solution_by_process(
    given_answers: List[str], ground_truths: List[str]
) -> bool:
    return False


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def validate_answer_format(passage: str) -> bool:
    if "[刑期]" not in passage and "[金额]" not in passage:
        return False
    if "<think>" not in passage or "</think>" not in passage:
        return False
    if "<answer>" not in passage or "</answer>" not in passage:
        return False
    return True


def extract_law_answer(passage: str) -> str:
    if "[刑期]" in passage:
        return passage.split("[刑期]")[-1].split("<eoa>", 1)[0].strip()
    elif "[金额]" in passage:
        return passage.split("[金额]")[-1].split("<eoa>", 1)[0].strip()


def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(
        given_answer, ground_truth
    )
