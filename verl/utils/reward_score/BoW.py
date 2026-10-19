from openai import OpenAI
import numpy as np

# the judge model served by vLLM (see README)
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


prompt_judge = lambda bow: f"""Task Instruction: Given a thought for the next word of a certain context, you need to predict this next word.
Now please give me your prediction of the next word based on the following thought:

{bow}

Next Word:"""
reference_judge = lambda context: f"""Given the following context:
{context}

What is the plausible immediate next word to this context? Please directly generate the immediate next word without any other words."""


def request_judge_model(messages, port, top_logprobs, temperature):
    # the judge continues the final assistant message; we read the top log-probs of its next token
    client = OpenAI(
        base_url=f'http://JUDGE_SERVER_IP:{port}/v1',
        api_key='bow'
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        max_tokens=3,
        temperature=temperature,
        extra_body={"continue_final_message": True, "add_generation_prompt": False, "top_logprobs": top_logprobs},
        logprobs=True
    )
    return response


def compute_score(solution_str, ground_truth, mode, soft_params=None):
    """
    ground_truth: the original completion for the prompt
    """
    if mode == "BoW-bottleneck":
        def get_tokens2probs(response_judge):
            tokens2probs = {'tokens': [], 'probs': []}
            for d in response_judge.choices[0].logprobs.content:
                tokens = []
                probs = []
                for c in d.top_logprobs:
                    tokens.append(c.token.strip())
                    probs.append(np.exp(c.logprob))
                tokens2probs['tokens'].append(tokens)
                tokens2probs['probs'].append(probs)
            return tokens2probs

        final = 0
        gold_word = ground_truth.split(' ')[0].strip()
        assert soft_params is not None
        _, judge_tokenizer, judge_port, judge_temperature, judge_top_k, context = soft_params

        encode_gold_word = judge_tokenizer.encode(gold_word, add_special_tokens=False)
        gold_word = judge_tokenizer.decode([encode_gold_word[0]]).strip() if len(encode_gold_word) > 1 else gold_word
        # use the judge model to get the two rewards for generated BoW.
        response_judge = request_judge_model(
            [{"role": "user", "content": prompt_judge(solution_str.strip())}, {"role": "assistant", "content": "Next Word:"}],
            port=judge_port,
            top_logprobs=judge_top_k,
            temperature=judge_temperature,
        )
        tokens2probs = get_tokens2probs(response_judge)

        judge_first_tokens = tokens2probs['tokens'][0]
        judge_first_probs = tokens2probs['probs'][0]
        if gold_word in judge_first_tokens:
            # add Reward-1
            final += judge_first_probs[judge_first_tokens.index(gold_word)]

        if context is not None: # this means using judge reference
            response_reference = request_judge_model(
                [{"role": "user", "content": reference_judge(context.strip())}, {"role": "assistant", "content": "The plausible immediate next word is:"}],
                port=judge_port,
                top_logprobs=judge_top_k,
                temperature=judge_temperature,
            )
            tokens2probs = get_tokens2probs(response_reference)

            reference_first_tokens = tokens2probs['tokens'][0]
            reference_first_probs = tokens2probs['probs'][0]
            reference_distance = 0.0
            for reference_token, reference_prob in zip(reference_first_tokens, reference_first_probs):
                if reference_token in judge_first_tokens:
                    reference_distance += abs(reference_prob - judge_first_probs[judge_first_tokens.index(reference_token)])
                else:
                    reference_distance += reference_prob
            final -= 0.1 * reference_distance
    else:
        raise NotImplementedError

    return final
