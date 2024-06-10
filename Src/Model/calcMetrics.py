import json
from transformers import AutoTokenizer
from Levenshtein import distance
from nltk.translate.bleu_score import sentence_bleu

def add_element_to_interval(element_id, element, intervals, interval_elements):
    interval_index = element_id // 100
    if interval_index >= len(intervals):
        print(element_id)
        print("out of range.")
        return

    interval = intervals[interval_index]
    interval_elements[interval].append(element)

def calculate_similarity(str1, str2):
    lev_distance = distance(str1, str2)
    similarity = 1 - lev_distance / max(len(str1), len(str2))
    return similarity


coderefine_path = "ModelRes/CodeRefine.json"
base_model_path = "ModelRes/BaseModels.json"
with open(coderefine_path) as f:
    coderefine = json.loads(f.read())
with open(base_model_path) as f:
    base_model = json.loads(f.read())

unopt_ir_2_truth_base_coderefine = {}
for unoptIR in coderefine:
    assert unoptIR in base_model
    unopt_ir_2_truth_base_coderefine[unoptIR] = {"coderefine_truth": coderefine[unoptIR]["coderefine_truth"], "coderefine": coderefine[unoptIR]["coderefine"]}
    unopt_ir_2_truth_base_coderefine[unoptIR]["base_truth"] = base_model[unoptIR]["base_truth"]
    unopt_ir_2_truth_base_coderefine[unoptIR]["codellama"] = base_model[unoptIR]["codellama"]
    unopt_ir_2_truth_base_coderefine[unoptIR]["codegemma"] = base_model[unoptIR]["codegemma"]

# check
for unopt_ir in unopt_ir_2_truth_base_coderefine:
    assert len(unopt_ir_2_truth_base_coderefine[unopt_ir]) == 5

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token_id = 2
tokenizer.padding_side = "left"

interval_metrics = {interval: None for interval in intervals}
# cnt = 0
for interval in interval_elements:
    coderefine_em = 0
    coderefine_bleu = 0
    coderefine_edit_sim = 0

    codellama_em = 0
    codellama_bleu = 0
    codellama_edit_sim = 0

    codegemma_em = 0
    codegemma_bleu = 0
    codegemma_edit_sim = 0

    for e in interval_elements[interval]:
        coderefine_truth = e['coderefine_truth']
        coderefine = e['coderefine']
        base_truth = e['base_truth']
        codellama = e['codellama']
        codegemma = e['codegemma']
        
        coderefine_em += coderefine_truth == coderefine
        coderefine_bleu += sentence_bleu([coderefine_truth.split(" ")], coderefine.split(" "))
        coderefine_edit_sim += calculate_similarity(coderefine_truth, coderefine)

        codellama_em += base_truth == codellama
        codellama_bleu += sentence_bleu([base_truth.split(" ")], codellama.split(" "))
        codellama_edit_sim += calculate_similarity(base_truth, codellama)

        codegemma_em += base_truth == codegemma
        codegemma_bleu += sentence_bleu([base_truth.split(" ")], codegemma.split(" "))
        codegemma_edit_sim += calculate_similarity(base_truth, codegemma)
    total = len(interval_elements[interval])
    if total == 0:
        break
    # cnt += coderefine_em
    interval_metrics[interval] = {"coderefine":{"em": coderefine_em/total, "bleu": coderefine_bleu/total, "editsim": coderefine_edit_sim/total},
                                  "codellama":{"em": codellama_em/total, "bleu": codellama_bleu/total, "editsim": codellama_edit_sim/total},
                                  "codegemma":{"em": codegemma_em/total, "bleu": codegemma_bleu/total, "editsim": codegemma_edit_sim/total}}

with open("metrics.json", "w") as f:
    json.dump(interval_metrics, f)