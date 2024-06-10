from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import argparse
import datasets
import shutil
import copy

def getPrompt(full_data):
    if infer_kind == "unopt2Opt":
        return "```".join(full_data.split("```")[:3])[:-1]
    elif infer_kind == "CodeRefine" or infer_kind == "CodeRefine-off-retriever":
        return "```".join(full_data.split("```")[:-2])[:-1]

def getCFG(cfg_ctx):
    BB2PredSuccs = {}
    cfg_ctx_list = cfg_ctx.split("\n")
    if len(cfg_ctx_list) % 3 != 0:
        return {}
    for i in range(0, len(cfg_ctx_list), 3):
        BBName = cfg_ctx_list[i][:-1]
        Preds = cfg_ctx_list[i+1].replace("Predecessors:", "").split(",")
        Succs = cfg_ctx_list[i+2].replace("Successors:", "").split(",")
        BB2PredSuccs[BBName] = {'pred': set(Preds), 'succ': set(Succs)}
    return BB2PredSuccs

def judge_correct(full_data, model_res):
    if "### Assistant" not in model_res:
        return False

    if len(model_res.split("```")[-2].split("\n"))<2:
        return False
    a = "\n".join(full_data.split("```")[-2].split("\n")[1:-1])
    b = "\n".join(model_res.split("```")[-2].split("\n")[1:-1])
    
    return a == b

def get_cfg(text):
    cfg_ctx = "\n".join(text.split("```")[7].split("\n")[1:]).rstrip().lstrip()
    cfg_ctx_list = cfg_ctx.split("\n")
    if len(cfg_ctx_list) % 3 != 0:
        return []
    bb_order = []
    for i in range(0, len(cfg_ctx_list), 3):
        BBName = cfg_ctx_list[i][:-1]
        bb_order.append(BBName.split("label ")[1])
    return bb_order

def get_opt_bb(text):
    res = "\n".join(text.split("```")[-2].split("\n")[1:]).rstrip().lstrip()
    bb_name = res.split("\n")[0]
    return "%"+bb_name[:-1], res

def collect_unoptIR(text):
    return "\n".join(text.split("```")[1].split("\n")[1:]).lstrip().rstrip()

def read_file(file_path):
    with open(file_path) as f:
        ctx = f.read()
    return ctx

def getAllFilesFromDir(dir):
    file_path_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            file_path_list.append(path)
    return file_path_list

def getPrompt(full_data):
    return "```".join(full_data.split("```")[:-2])[:-1]

def concat(test_dataset, tokenizer, out_truth_dir, wrong_res, out_unopt_dir, out_coderefine_res_dir):
    unopt_2_groud_truth = {}
    unopt_2_func_head = {}
    for d in test_dataset:
        ir = collect_unoptIR(d['text'])
        if ir not in unopt_2_groud_truth:
            bbs = get_cfg(d['text'])
            unopt_2_groud_truth[ir] = {x: None for x in bbs}
            unopt_2_func_head[ir] = collect_unoptIR(d['text']).split("\n")[0]
        bb_name, optbb = get_opt_bb(d['text'])
        unopt_2_groud_truth[ir][bb_name] = optbb
    
    # check
    for ir in unopt_2_groud_truth:
        for bb in unopt_2_groud_truth[ir]:
            assert unopt_2_groud_truth[ir][bb] != None
    
    # write truth to file
    for idx, ir in enumerate(unopt_2_groud_truth):
        with open(f"{out_truth_dir}/{idx}.ll", "w") as f:
            f.write(unopt_2_func_head[ir]+"\n")
            for bb in unopt_2_groud_truth[ir]:
                f.write(unopt_2_groud_truth[ir][bb]+"\n")
            f.write("}")
    
    ir_2_wrong_bb = copy.deepcopy(unopt_2_groud_truth)
    for file_path in getAllFilesFromDir(wrong_res):
        if file_path.endswith(".truth.ll"):
            continue
        ctx = read_file(file_path)
        ir = collect_unoptIR(ctx)
        bb_name, wrongbb = get_opt_bb(ctx)
        assert ir in ir_2_wrong_bb
        assert bb_name in ir_2_wrong_bb[ir]
        ir_2_wrong_bb[ir][bb_name] = wrongbb
    
    # when over the limit, cannot generate the refined segment
    for d in test_dataset:
        eval_prompt = getPrompt(d['text'])
        if len(tokenizer(eval_prompt)['input_ids']) > 2500:
            ir = collect_unoptIR(d['text'])
            bb_name, wrongbb = get_opt_bb(d['text'])
            assert ir in ir_2_wrong_bb
            assert bb_name in ir_2_wrong_bb[ir]
            ir_2_wrong_bb[ir][bb_name] = ""
    
    # write the unrefined version to file
    for idx, ir in enumerate(ir_2_wrong_bb):
        with open(f"{out_unopt_dir}/{idx}.ll", "w") as f:
            f.write(ir)

    # write wrong to file
    for idx, ir in enumerate(ir_2_wrong_bb):
        with open(f"{out_coderefine_res_dir}/{idx}.ll", "w") as f:
            f.write(unopt_2_func_head[ir]+"\n")
            for bb in ir_2_wrong_bb[ir]:
                f.write(ir_2_wrong_bb[ir][bb]+"\n")
            f.write("}")
#############Parse Arg#############
parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('-m','--model')
parser.add_argument('-token-num','--token_num')
parser.add_argument('-infer-kind','--infer_kind')
parser.add_argument('-test-dataset-dir', '--test_dataset_dir')
parser.add_argument('-adapters-dir', '--adapters_dir')
parser.add_argument('-output-dir', '--output_dir')


args = parser.parse_args()
base_model = args.model
token_num = args.token_num
infer_kind = args.infer_kind
test_dataset_path = args.test_dataset_dir
adapters_dir = args.adapters_dir
output_dir = args.output_dir

#############Prepare for infering#############
test_dataset = datasets.load_from_disk(test_dataset_path)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapters_dir)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = 2
tokenizer.padding_side = "left"

# create dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(f"{output_dir}/WrongSegment"):
    os.makedirs(f"{output_dir}/WrongSegment")
else:
    shutil.rmtree(f"{output_dir}/WrongSegment")
    os.makedirs(f"{output_dir}/WrongSegment")

#############Infer#############
acc = 0
too_long = 0
for idx, item in enumerate(test_dataset):
    # if idx % 100 == 0:
    print(f"idx:{idx}\tacc_num:{acc}\tmore than ctx:{too_long}")
    eval_prompt = getPrompt(item['text'])
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    if len(tokenizer(eval_prompt)['input_ids']) > token_num:
        too_long += 1
        continue
    model_res = tokenizer.decode(model.generate(**model_input, max_new_tokens=token_num, pad_token_id=tokenizer.eos_token_id)[0])
    # print(model_res)
    if len(model_res.split("```")) <= 3:
        print("idx:", idx, "\n", model_res)
        continue
    if judge_correct(item['text'], model_res):
        acc += 1
    else:
        #print wrong ans
        print("Write Wrong...")
        with open(f"{output_dir}/WrongSegment/{idx}.model.predict.ll", "w") as f:
            f.write(model_res)
        with open(f"{output_dir}/WrongSegment/{idx}.truth.ll", "w") as f:
            f.write(item['text'])
    # break

print(f"accuracy: {acc/len(test_dataset)}")

#############Concat#############
if "CodeRefine" in infer_kind:
    concat(test_dataset, tokenizer, f"{output_dir}/truth", f"{output_dir}/WrongSegment", f"{output_dir}/unrefined", f"{output_dir}/InferRes")
