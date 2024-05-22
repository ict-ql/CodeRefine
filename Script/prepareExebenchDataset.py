import os
import argparse
from transformers import AutoTokenizer
import random
import datasets
from random import shuffle

# dataset_dir = "/home/ql/SRC/ActionModel/Dataset/ComPileC++"
# train_ratio = 0.8
# token_num = 9000
# out_dir = f"/home/ql/SRC/ActionModel/Dataset/ComPileC++-filter-{token_num}"
base_model = "codellama/CodeLlama-7b-Instruct-hf"

os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"

def gen_new_prompt(input_text, white_list_key_order):
    ll = input_text.split("```")
    kind_2_res = {}
    assert len(ll) % 2 == 1, input_text
    for i in range(0, len(ll), 2):
        if i+1 >= len(ll):
            continue
        kind_2_res[ll[i]] = ll[i+1]
    new_prompt = []
    for key in white_list_key_order:
        new_prompt.append(key)
        new_prompt.append(kind_2_res[key])
    new_prompt.append("\n")
    res = "```".join(new_prompt).replace("\n\n", "\n")
    if res.startswith("\n"):
        res = res[1:]
    return res

def filter_origin_dataset_with_only_focus_on_keyword_and_gen_new_one(dataset, output_dir, white_list_key_order):
    new_prompts = []
    for data in dataset:
        new_prompt = gen_new_prompt(data['text'], white_list_key_order)
        new_prompts.append(new_prompt)
    dictt = {"text": new_prompts}
    new_dataset = datasets.arrow_dataset.Dataset.from_dict(dictt)
    new_dataset.save_to_disk(output_dir)

def filter_dataset_to_train_eval_dataset(dataset, tokenizer, token_num, train_ratio):
    res_after_filter = []
    for data in dataset:
        if token_num != None:
            data_tokenize_res = tokenizer(data['text'])
            if len(data_tokenize_res['input_ids']) > int(token_num):
                continue
        res_after_filter.append(data['text'])
    random.shuffle(res_after_filter)
    train_datas = res_after_filter[:int(len(res_after_filter)*train_ratio)]
    eval_datas = res_after_filter[int(len(res_after_filter)*train_ratio):]
    train_dict = {"text": train_datas}
    train_dataset = datasets.arrow_dataset.Dataset.from_dict(train_dict)

    eval_dict = {"text": eval_datas}
    eval_dataset = datasets.arrow_dataset.Dataset.from_dict(eval_dict)

    return train_dataset, eval_dataset

def merge_dataset(dataset_dirs):
    dds = []
    for ddir in dataset_dirs:
        dds.append(datasets.load_from_disk(ddir))
    res = datasets.concatenate_datasets(dds)
    return res

def collect_unoptIR(text):
    return "\n".join(text.split("```")[1].split("\n")[1:]).lstrip().rstrip()

def collect_unoptIR_in_dataset(dataset):
    unoptIRs = set()
    for d in dataset:
        unoptIR = collect_unoptIR(d['text'])
        unoptIRs.add(unoptIR)
    return unoptIRs

def split_set(s, ratio):
    lst = list(s)
    idx = int(len(lst) * ratio)
    return set(lst[:idx]), set(lst[idx:])

def calc_total_elem_num(a, b):
    return len(a) + len(b) - len(a&b)

def gen_train_and_eval_by_ratio(s_train_unoptIRs, s_eval_unoptIRs, train_ratio):
    train_data_num = int(calc_total_elem_num(s_train_unoptIRs, s_eval_unoptIRs)*train_ratio)
    union_elems = s_train_unoptIRs & s_eval_unoptIRs
    s_train_unoptIRs_pure = s_train_unoptIRs - union_elems
    train_extra_elems_num = train_data_num - len(s_train_unoptIRs_pure)
    ratio = train_extra_elems_num / len(union_elems)
    train_elements, eval_elements = split_set(union_elems, ratio)
    s_train_unoptIRs = s_train_unoptIRs - eval_elements
    s_eval_unoptIRs = s_eval_unoptIRs - train_elements
    return s_train_unoptIRs, s_eval_unoptIRs

def get_prompt_including_target_unoptIRs(dataset, targets):
    prompts = set()
    for d in dataset:
        unoptIR = collect_unoptIR(d['text'])
        if unoptIR in targets:
            prompts.add(d['text'])
    return prompts

def gen_train_eval_dataset_correspongding_standard_dataset(s_train_dataset, s_eval_dataset, t_dataset, train_ratio):
    s_train_unoptIRs = collect_unoptIR_in_dataset(s_train_dataset)
    s_eval_unoptIRs = collect_unoptIR_in_dataset(s_eval_dataset)
    s_train_unoptIRs, s_eval_unoptIRs = gen_train_and_eval_by_ratio(s_train_unoptIRs, s_eval_unoptIRs, train_ratio)
    assert len(s_train_unoptIRs & s_eval_unoptIRs) == 0
    
    t_train_prompt = get_prompt_including_target_unoptIRs(t_dataset, s_train_unoptIRs)
    t_eval_prompt = get_prompt_including_target_unoptIRs(t_dataset, s_eval_unoptIRs)

    train_dict = {"text": t_train_prompt}
    train_dataset = datasets.arrow_dataset.Dataset.from_dict(train_dict)

    eval_dict = {"text": t_eval_prompt}
    eval_dataset = datasets.arrow_dataset.Dataset.from_dict(eval_dict)
    return train_dataset, eval_dataset

def gen_dataset_correspongding_standard_dataset(s_dataset, t_dataset):
    s_unoptIRs = collect_unoptIR_in_dataset(s_dataset)
    t_prompt = get_prompt_including_target_unoptIRs(t_dataset, s_unoptIRs)
    t_dict = {"text": t_prompt}
    dataset = datasets.arrow_dataset.Dataset.from_dict(t_dict)
    return dataset

def delete_retriver_in_prompt(text):
    tmp = text.split("```")
    res = []
    res.extend(tmp[:2])
    res.extend(tmp[4:8])
    res.extend(tmp[-3:-1])
    return "```".join(res)

def gen_no_retriever_dataset(origin_dataset):
    prompts = []
    for d in origin_dataset:
        prompts.append(delete_retriver_in_prompt(d['text']))
    dictt = {"text": prompts}
    dataset = datasets.arrow_dataset.Dataset.from_dict(dictt)
    return dataset

def get_dataset_include_irs(origin_dataset, irs):
    prompts = []
    for d in origin_dataset:
        unoptIR = collect_unoptIR(d['text'])
        if unoptIR in irs:
            prompts.append(d['text'])
    dictt = {"text": prompts}
    dataset = datasets.arrow_dataset.Dataset.from_dict(dictt)
    return dataset

parser = argparse.ArgumentParser(description='Gen Dataset')
parser.add_argument('-m','--mode')
parser.add_argument('-i','--input_dir')
parser.add_argument('-o','--output_dir')

parser.add_argument('-standard-dir', '--standard_dir')
parser.add_argument('-target-dir', '--target_dir')

parser.add_argument('-token-num','--token_num')
parser.add_argument('-train-ratio','--train_ratio')

parser.add_argument('-unopt2opt-dir', '--unopt2opt_dir')
parser.add_argument('-cfg-dir', '--cfg_dir')
parser.add_argument('-bb-dir', '--bb_dir')

args = parser.parse_args()

if args.mode == "filter":
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = datasets.load_from_disk(args.input_dir)
    # assert args.token_num, f"token num:`{args.token_num}` invalid!\n"
    assert args.train_ratio, f"train ratio:`{args.train_ratio}` invalid!\n"
    train_dataset, eval_dataset = filter_dataset_to_train_eval_dataset(dataset, tokenizer, args.token_num, float(args.train_ratio))
    train_dataset.save_to_disk(f"{args.output_dir}/train_dataset")
    eval_dataset.save_to_disk(f"{args.output_dir}/eval_dataset")
elif args.mode == "merge":
    dataset_dirs = args.input_dir.split(",")
    assert len(dataset_dirs) > 1, f"the {args.input_dir} must have more than one dataset!\n"
    res = merge_dataset(dataset_dirs)
    res.save_to_disk(args.output_dir)
elif args.mode == "gen-bb-prompt":
    white_list_key_order = ['### Instruction: optimize the following IR with O3\n### LLVM IR:\n', '\n### attributes:\n', '\n\n### Assistant:\n']
    dataset = datasets.load_from_disk(args.input_dir)
    filter_origin_dataset_with_only_focus_on_keyword_and_gen_new_one(dataset, args.output_dir, white_list_key_order)
elif args.mode == "gen-func-prompt":
    white_list_key_order = ['\n\n### unoptFunc:\n', '\n\n### optFunc:\n']
    dataset = datasets.load_from_disk(args.input_dir)
    filter_origin_dataset_with_only_focus_on_keyword_and_gen_new_one(dataset, args.output_dir, white_list_key_order)
elif args.mode == "gen-baseline-dataset":
    assert args.train_ratio, f"train ratio:`{args.train_ratio}` invalid!\n"
    assert args.standard_dir and args.target_dir, f"standard dir:`{args.standard_dir}` or target dir:`{args.target_dir}` invalid!\n"
    s_train_dataset = datasets.load_from_disk(f"{args.standard_dir}/train_dataset")
    s_eval_dataset = datasets.load_from_disk(f"{args.standard_dir}/eval_dataset")
    t_dataset = datasets.load_from_disk(f"{args.target_dir}")
    train_dataset, eval_dataset = gen_train_eval_dataset_correspongding_standard_dataset(s_train_dataset, s_eval_dataset, t_dataset, float(args.train_ratio))
    train_dataset.save_to_disk(f"{args.output_dir}/train_dataset")
    eval_dataset.save_to_disk(f"{args.output_dir}/eval_dataset")
elif args.mode == "gen-datatset-from-input":
    s_dataset = datasets.load_from_disk(f"{args.standard_dir}")
    t_dataset = datasets.load_from_disk(f"{args.target_dir}")
    dataset = gen_dataset_correspongding_standard_dataset(s_dataset, t_dataset)
    dataset.save_to_disk(f"{args.output_dir}")
elif args.mode == "gen-no-retriever-datatset":
    train_dataset = datasets.load_from_disk(f"{args.input_dir}/train_dataset")
    eval_dataset = datasets.load_from_disk(f"{args.input_dir}/eval_dataset")
    new_train_dataset = gen_no_retriever_dataset(train_dataset)
    new_eval_dataset = gen_no_retriever_dataset(eval_dataset)
    new_train_dataset.save_to_disk(f"{args.output_dir}/train_dataset")
    new_eval_dataset.save_to_disk(f"{args.output_dir}/eval_dataset")
elif args.mode == "gen-no-retriever-datatset-infer":
    dataset = datasets.load_from_disk(f"{args.input_dir}")
    new_dataset = gen_no_retriever_dataset(dataset)
    new_dataset.save_to_disk(f"{args.output_dir}")
elif args.mode == "gen-all-infer-dataset":
    unopt2Opt_dataset = datasets.load_from_disk(f"{args.unopt2opt_dir}")

    sample_num = 1200 #多抽几个，然后取获得大家都有的一个set，然后再随机从里面选1000个
    indices = random.sample(range(len(unopt2Opt_dataset)), k=sample_num)
    sample_unopt2Opt_dataset = unopt2Opt_dataset.select(indices)
    unoptIRs = collect_unoptIR_in_dataset(sample_unopt2Opt_dataset)

    t_dataset = datasets.load_from_disk(f"{args.cfg_dir}")
    cfg_dataset = gen_dataset_correspongding_standard_dataset(sample_unopt2Opt_dataset, t_dataset)
    unoptIRs = unoptIRs & collect_unoptIR_in_dataset(cfg_dataset)

    t_dataset = datasets.load_from_disk(f"{args.bb_dir}")
    bb_dataset = gen_dataset_correspongding_standard_dataset(sample_unopt2Opt_dataset, t_dataset)
    unoptIRs = unoptIRs & collect_unoptIR_in_dataset(bb_dataset)
    
    unoptIRs = random.sample(list(unoptIRs), 1000)
    sample_unopt2Opt_dataset = get_dataset_include_irs(sample_unopt2Opt_dataset, unoptIRs)
    cfg_dataset = get_dataset_include_irs(cfg_dataset, unoptIRs)
    bb_dataset = get_dataset_include_irs(bb_dataset, unoptIRs)

    print(len(collect_unoptIR_in_dataset(sample_unopt2Opt_dataset)), len(collect_unoptIR_in_dataset(cfg_dataset)), len(collect_unoptIR_in_dataset(bb_dataset)))
    
    sample_unopt2Opt_dataset.save_to_disk(f"{args.output_dir}/full-unopt2Opt-origin-infer")
    cfg_dataset.save_to_disk(f"{args.output_dir}/full-unopt2optCFG-infer")
    bb_dataset.save_to_disk(f"{args.output_dir}/full-unopt2Optbb-infer")
