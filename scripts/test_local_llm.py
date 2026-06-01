from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "local_models" / "Qwen2.5-0.5B-Instruct"


def main():
    print(f"Loading local model from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个电影推荐助手，只输出简洁中文。"
        },
        {
            "role": "user",
            "content": "帮我把这个需求改写成电影检索关键词：推荐5部2010年之后的高评分科幻惊悚片，不要恐怖片。"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print("\n===== Local LLM Response =====")
    print(response)


if __name__ == "__main__":
    main()
