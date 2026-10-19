# load the BoW prompts from the pre-training corpus
import os
import datasets
import json
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load('en_core_web_sm')
spacy_tokenizer = Tokenizer(nlp.vocab)

# inputs (detective-puzzles.csv, gpt4.1mini_filter_info.json) and the default output folder are resolved
# from the repo root, so this script can be run from any directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


thought_prompt_v1 = lambda x: f"""Task Instruction: You are provided with a specific context and must generate a concise reasoning describing the possible immediate next words. Often, multiple words could fit naturally, making it impossible to determine one definite next word. Your task is to provide a comprehensive, neutral, and unbiased explanation of the types or categories of words that could reasonably follow. Ensure your reasoning is fully self-contained, such that another person, without access to the original context, can understand your logic and clearly grasp what kinds of words are appropriate next choices.
    
Now please give me your prediction for the thought of the next word based on the following context:

{x}

Thought:"""
thought_prompt_v2 = lambda x: f"""The next word prediction problem of language modeling generally involves first searching for a comprehensive but small enough set of possible next words, and then estimating each possible word's probability within this set. Given a specific context, your job is to output a reasoning process that generally follows this illustration.

Here is one important rule you MUST follow. Since different candidate sets are different in size, instead of enumerating all possible words, you must be smart and innovative enough to describe the word set in concise and unbiased natural language, without missing any possibilities. This is particularly important when the smallest candidate set you can find is still not small or concrete enough. 
For example, given the context "I [next-word]", the best you can do is to tell me that the next word set will likely contain verbs, and those verbs should be actions performed by humans rather than other verbs. You should not give a concrete list of words since it is impossible to enumerate all the verbs humans can perform. 
However, in other contexts where information is rich, you should perform more in-depth reasoning. For example, given the context "I like fruit, so as lunch I had two [next-word]", you should reason that the next word set will likely contain nouns of fruit given the logical clue that "I like fruit" and the quantitative adjective "two" in the context. Furthermore, you should reason that this set is more likely to contain larger fruits to fill me up for lunch instead of small fruit. 
In other scenarios, such as "1+1=[next-word]", where the next word set can be small enough to be listed out, you should actually list out all possible words in the set you find. 

Now, given the below context, please reason on possible next words following all the rules described above. Make your output self-contained without needing to reference the given context.
Context:
{x}"""

thought_prompt_nojudge = lambda x: f"""The next word prediction problem of language modeling generally involves first searching for a comprehensive but small enough set of possible next words, and then estimating each possible word's probability within this set to pick the most likely next word. Given a specific context, your job is to output a reasoning process that generally follows this illustration to predict the most likely next word.

Here is one important rule you MUST follow. Since different candidate sets are different in size, instead of enumerating all possible words, you must be smart and innovative enough to describe the word set in concise and unbiased natural language, without missing any possibilities. This is particularly important when the smallest candidate set you can find is still not small or concrete enough.
For example, given the context "I like fruit, so as lunch I had two [next-word]", you should reason that the next word set will likely contain nouns of fruit given the logical clue that "I like fruit" and the quantitative adjective "two" in the context. Furthermore, you should reason that this set is more likely to contain larger fruits to fill me up for lunch instead of small fruit. Then you should choose the most likely next word based on the set description.

Now, given the below context, please follow the rule described above to first reason and then wrap your predicted most likely next word within \\boxed{{}}. Make your output self-contained without needing to reference the given context.
Context:
{x}"""


def train_mystery_all_make_map_fn(split, use_instruct=False, use_filter=False, instruction_version='v2', data_source='BoW-mystery'):
    # load the gpt4.1mini filter once, not once per story
    id_to_require_reasoning = None
    if use_filter:
        with open(os.path.join(REPO_ROOT, 'gpt4.1mini_filter_info.json'), 'r') as f:
            id_to_require_reasoning = json.load(f)

    def process_fn(examples, ids):
        data = defaultdict(list)
        for case_name, raw_doc, idx in zip(examples['case_name'], examples['mystery_text'], ids):
            seperated_spacy_tokens = [token for token in spacy_tokenizer(raw_doc)]
            separated_tokens = [token.text for token in seperated_spacy_tokens]

            for i in range(1, len(separated_tokens)):
                
                if use_filter: # this is for applying gpt4.1mini filtering
                    if not id_to_require_reasoning.get(case_name+"_"+str(i), False):
                        continue

                raw_prompt = ' '.join(separated_tokens[:i])  
                completion = ' '.join(separated_tokens[i:])

                if use_instruct:
                    # add instruct prompts for vanilla instruct models
                    if instruction_version == 'v1':
                        prompt = [{"role": "user", "content": thought_prompt_v1(raw_prompt)}]
                    elif instruction_version == 'v2':
                        prompt = [{"role": "user", "content": thought_prompt_v2(raw_prompt)}]
                    elif instruction_version == 'nojudge':
                        prompt = [{"role": "user", "content": thought_prompt_nojudge(raw_prompt)}]
                    elif instruction_version == 'inferior':
                        raise ValueError(f"Invalid instruction version: {instruction_version}")
                else:
                    raise NotImplementedError("Not implemented for non-instruct models")

                data['id'].append(case_name+"_"+str(i))
                data['raw_prompt'].append(raw_prompt)
                data['raw_next_word'].append(separated_tokens[i:][0])
                data['case_name'].append(case_name)
                
                data['raw_text'].append(raw_doc)
                data['data_source'].append(data_source)
                data['prompt'].append(prompt)
                data['ability'].append("bow")
                data['reward_model'].append({
                    "style": "rule",
                    "ground_truth": completion
                })
                data['extra_info'].append({
                    'split': split,
                    'index': f"{idx}_{i}",
                    'answer': completion,
                    "question": raw_prompt,
                })

        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=os.path.join(REPO_ROOT, 'data', 'bow'))
    args = parser.parse_args()

    # the < 2048-token story filter below uses this tokenizer; changing it changes which stories are kept
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    train_data_path = REPO_ROOT
    train_mystery = {"train": "detective-puzzles.csv"}

    dataset = datasets.load_dataset("csv", data_dir=train_data_path, data_files=train_mystery)
    train_dataset_mystery_all_original = dataset['train'].filter(lambda x: len(tokenizer.encode(x['mystery_text'])) < 2048)

    train_dataset_mysteryAll_bottleneck_thoughtv2_gpt41minifiltered = train_dataset_mystery_all_original.map(function=train_mystery_all_make_map_fn('train', data_source='BoW-bottleneck', use_instruct=True, use_filter=True, instruction_version='v2'), with_indices=True, batched=True, remove_columns=train_dataset_mystery_all_original.column_names)
    print(len(train_dataset_mysteryAll_bottleneck_thoughtv2_gpt41minifiltered))

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset_mysteryAll_bottleneck_thoughtv2_gpt41minifiltered.to_parquet(os.path.join(local_dir, 'mystery_all_bow-bottleneck-thoughtv2-instruct_train_gpt4.1minifiltered.parquet'))
