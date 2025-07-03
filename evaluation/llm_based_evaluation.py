import argparse
import json
import re
import time

# import openai
import traceback
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from prompts.RAL_evaluator import EVALUATION_PROMPT
from tqdm import tqdm


def load_jsonl(file_path):
    _data = []
    with open(file_path, "r") as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def get_payload(line):
    instruction = line["instruction"][:6000]
    question = line["question"]
    if line["output"] is not None:
        output = line["output"][:4000]
    else:
        output = "None"
    content = SYS_MSG.format(input=instruction, output=output, question=question)
    payload = {
        "model": args.eval_model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 0.3,
        "stream": True,
    }
    return payload


def save_jsonl(entry, sava_path):
    with open(sava_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_answer(input_data: dict, retry=30):
    entry, save_path = input_data["data"], input_data["save_path"]
    try:
        # payload = get_payload(entry)
        # chat_completion = openai.ChatCompletion.create(model=payload['model'], temperature=0, messages=payload['messages'])
        # generation = chat_completion.choices[0].message.content

        payload = get_payload(entry)
        response = client.chat.completions.create(
            model=payload["model"], temperature=0, messages=payload["messages"]
        )
        # response = client.chat.completions.create(**payload)
        generation = response.choices[0].message.content

        if generation is None or generation.strip() == "":
            return get_answer(input_data, retry=retry - 1)

        re_result = re.findall(r"答案：是|答案：否", generation)
        if len(re_result) == 1:
            if "是" in re_result[0]:
                entry["point_judge"] = True
            else:
                entry["point_judge"] = False
        else:
            if "是" in generation and "否" not in generation:
                entry["point_judge"] = True
            else:
                entry["point_judge"] = False

        entry["point_explanation"] = generation
        entry["payload"] = payload
        save_jsonl(entry, save_path)
        return entry
    except Exception as e:
        time.sleep(1.2)
        retry -= 1
        if retry < 0:
            entry["point_judge"] = False
            entry["point_explanation"] = "None"
            entry["payload"] = payload
            save_jsonl(entry, save_path)
            return entry
        print(f"retry:剩余{retry}次")
        print(e)
        print(traceback.format_exc())
        return get_answer(input_data, retry=retry)


def run_evaluation(save_path, datas, num_pool):
    _input = [
        {"data": i, "eval_model": args.eval_model_name, "save_path": save_path}
        for i in datas
        if i
    ]
    with ThreadPoolExecutor(max_workers=num_pool) as executor:
        tqdm(
            executor.map(get_answer, _input),
            total=len(_input),
            desc="Processing",
            ncols=100,
        )


def get_data(data_path, llm_output_path, language="zh"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(llm_output_path, "r", encoding="utf-8") as f:
        outputs = [json.loads(line) for line in f.readlines()]

    datas = []
    for i, (d, o) in enumerate(zip(data, outputs)):
        for j, q in enumerate(d["scoring_questions"]):
            if q["rule"] is not None:
                continue

            if language == "zh":
                datas.append(
                    {
                        "main_id": i,
                        "point_id": j,
                        "instruction": d["instruction"],
                        "question": q["question"],
                        "output": o["generated"],
                    }
                )
            elif language == "en":
                datas.append(
                    {
                        "main_id": i,
                        "point_id": j,
                        "instruction": d["instruction_en"],
                        "question": q["question_en"],
                        "output": o["generated"],
                    }
                )
    return datas


def main_run(args):
    datas = get_data(
        data_path=args.data_path,
        llm_output_path=args.llm_output_path,
        language=args.language,
    )
    run_evaluation(args.output_path, datas, args.num_pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--llm_output_path", type=str, default="")
    parser.add_argument("--num_pool", type=int, default=40)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--eval_model_name", type=str, default="Azure/gpt-4o")
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()
    # openai.api_key = args.api_key
    # openai.api_base = args.api_base
    SYS_MSG = EVALUATION_PROMPT

    global client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    main_run(args)
