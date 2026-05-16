import json
import os
import re

import tiktoken

from auto_scoring_judge import AutoScoringJudge
from parser_helper import is_equiv, last_boxed_only_string, remove_boxed
from parsers import Parser, test_solution

scorer = AutoScoringJudge()


def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def extract_steps_decoded(item):
    if "steps_decoded" in item and item["steps_decoded"] is not None:
        try:
            return int(item["steps_decoded"])
        except (TypeError, ValueError):
            return None

    confidence_logs = item.get("confidence_logs")
    if isinstance(confidence_logs, dict):
        step_counts = confidence_logs.get("step_decoded_counts")
        if isinstance(step_counts, list):
            return len(step_counts)

    return None


def _load_generation_data(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            return json.load(file)
    return json_data


def parse_gsm_answers(json_path=None, json_data=None):
    data = _load_generation_data(json_path=json_path, json_data=json_data)
    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens
        steps_decoded = extract_steps_decoded(item)

        parsed_answer = None
        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
                "steps_decoded": steps_decoded,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_math_answers(json_path=None, json_data=None):
    data = _load_generation_data(json_path=json_path, json_data=json_data)
    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens
        steps_decoded = extract_steps_decoded(item)

        parsed_answer = None
        try:
            parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        except Exception:
            parsed_answer = None

        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                parsed_answer = answer_match.group(1).strip()

        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, ground_truth)

        if not is_correct:
            is_correct = scorer.judge(ground_truth, parsed_answer, precision=1e-6)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
                "steps_decoded": steps_decoded,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_countdown_answers(json_path=None, json_data=None):
    data = _load_generation_data(json_path=json_path, json_data=json_data)
    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    def validate_equation(equation_str, available_numbers):
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            return sorted(numbers_in_eq) == sorted(available_numbers)
        except Exception:
            return False

    def evaluate_equation(equation_str):
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            return eval(equation_str.strip(), {"__builtins__": None}, {})
        except Exception:
            return float("Inf")

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        generated_text = item.get("generations", "")

        effective_tokens = count_effective_tokens(generated_text)
        total_effective_tokens += effective_tokens
        steps_decoded = extract_steps_decoded(item)

        numbers = []
        target = None
        if isinstance(ground_truth, list) and len(ground_truth) == 2:
            numbers = ground_truth[0]
            target = ground_truth[1]
        else:
            numbers_match = re.search(r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE)
            if numbers_match:
                numbers = [int(num.strip()) for num in numbers_match.group(1).split(",")]

            target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
            if target_match:
                target = int(target_match.group(1))

        try:
            equation = remove_boxed(last_boxed_only_string(generated_text))
        except Exception:
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            equation = answer_match.group(1).strip() if answer_match else generated_text

        equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
        equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
        if equation_match:
            equation = equation_match.group(1).strip()

        is_correct = False
        result = None
        if validate_equation(equation, numbers):
            result = evaluate_equation(equation)
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
                total_correct += 1

        processed_items.append(
            {
                "question": question,
                "extracted_answer": equation,
                "evaluation_result": result,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
                "steps_decoded": steps_decoded,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_sudoku_answers(json_path=None, json_data=None):
    data = _load_generation_data(json_path=json_path, json_data=json_data)
    total_correct_cells = 0
    total_empty_cells = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens
        steps_decoded = extract_steps_decoded(item)

        puzzle_str = ""
        if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
            puzzle_str = question[:16]
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)

        assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"

        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)
        solution_str = ""
        patterns = [
            r"<answer>.*?```\s*([\d\s]+)```",
            r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
            r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
            r".*?(\d{16})\s*</answer>",
            r"\b(\d{16})\b",
        ]

        for pattern in patterns:
            if solution_str:
                break
            match = re.search(pattern, raw_generation, re.DOTALL)
            if match and match.group(1).strip():
                solution_str = match.group(1).strip()

        solution_str = re.sub(r"\s", "", solution_str)
        if not solution_str:
            correct_cells = 0
        else:
            if len(solution_str) < 16:
                solution_str = solution_str + "0" * (16 - len(solution_str))
            elif len(solution_str) > 16:
                solution_str = solution_str[:16]
            correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])

        accuracy = correct_cells / empty_cells if empty_cells > 0 else 0.0
        total_correct_cells += correct_cells
        total_empty_cells += empty_cells

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": solution_str,
                "ground_truth": ground_truth,
                "empty_cells": empty_cells,
                "correct_cells": correct_cells,
                "accuracy": accuracy,
                "effective_tokens": effective_tokens,
                "steps_decoded": steps_decoded,
            }
        )

    return total_correct_cells, total_empty_cells, processed_items, total_effective_tokens * 8


def parse_humaneval_answers(json_path=None, json_data=None, output_dir=None):
    data = _load_generation_data(json_path=json_path, json_data=json_data)
    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens
        steps_decoded = extract_steps_decoded(item)

        program = Parser.extract_answer_code(raw_generation)
        test_code = None
        is_correct = False
        error_type = None

        if program is not None:
            try:
                solution_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", program)
                if solution_match:
                    defined_func = solution_match.group(1)
                    unit_test = ground_truth[ground_truth.index("def") :]
                    test_code = program + "\n\n" + unit_test + "\n\n" + f"check({defined_func})"
                    is_correct = bool(test_solution(test_code, output_dir))
                    if is_correct:
                        total_correct += 1
                else:
                    error_type = "no_function_found"
            except Exception as e:
                error_type = f"execution_build_error: {type(e).__name__}"
        else:
            error_type = "no_code_extracted"

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": program,
                "ground_truth": ground_truth,
                "test_code": test_code,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
                "error_type": error_type,
                "steps_decoded": steps_decoded,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_mbpp_answers(json_path=None, json_data=None, output_dir=None):
    data = _load_generation_data(json_path=json_path, json_data=json_data)
    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens
        steps_decoded = extract_steps_decoded(item)

        program = Parser.extract_answer_code(raw_generation)
        test_code = None
        is_correct = False
        error_type = None

        if program is not None:
            try:
                test_code = program + "\n\n" + ground_truth
                is_correct = bool(test_solution(test_code, output_dir))
                if is_correct:
                    total_correct += 1
            except Exception as e:
                error_type = f"execution_build_error: {type(e).__name__}"
        else:
            error_type = "no_code_extracted"

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": program,
                "ground_truth": ground_truth,
                "test_code": test_code,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
                "error_type": error_type,
                "steps_decoded": steps_decoded,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def score_generations(dataset_name, generations, output_dir=None):
    json_data = {"generations": generations}
    dataset_name = dataset_name.lower()
    if dataset_name in {"humaneval", "mbpp"} and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if "gsm" in dataset_name:
        correct, processed, detailed_results, total_effective_tokens = parse_gsm_answers(json_data=json_data)
    elif "math" in dataset_name or "aime" in dataset_name:
        correct, processed, detailed_results, total_effective_tokens = parse_math_answers(json_data=json_data)
    elif "countdown" in dataset_name:
        correct, processed, detailed_results, total_effective_tokens = parse_countdown_answers(json_data=json_data)
    elif "sudoku" in dataset_name:
        correct, processed, detailed_results, total_effective_tokens = parse_sudoku_answers(json_data=json_data)
    elif "humaneval" in dataset_name:
        correct, processed, detailed_results, total_effective_tokens = parse_humaneval_answers(
            json_data=json_data,
            output_dir=output_dir,
        )
    elif "mbpp" in dataset_name:
        correct, processed, detailed_results, total_effective_tokens = parse_mbpp_answers(
            json_data=json_data,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unsupported dataset for scoring: {dataset_name}")

    total_steps_decoded = 0
    steps_observed = 0
    for item in detailed_results:
        steps_decoded = item.get("steps_decoded")
        if steps_decoded is not None:
            total_steps_decoded += int(steps_decoded)
            steps_observed += 1

    accuracy = (correct / processed * 100) if processed > 0 else 0.0
    avg_effective_tokens = (total_effective_tokens / processed) if processed > 0 else 0.0
    avg_steps_decoded = (total_steps_decoded / steps_observed) if steps_observed > 0 else None

    return {
        "correct": correct,
        "processed": processed,
        "accuracy": accuracy,
        "total_effective_tokens": total_effective_tokens,
        "avg_effective_tokens": avg_effective_tokens,
        "total_steps_decoded": total_steps_decoded,
        "steps_observed": steps_observed,
        "avg_steps_decoded": avg_steps_decoded,
        "detailed_results": detailed_results,
    }
