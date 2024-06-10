import argparse
import json
import re
import InteractionWithShell as shell
import threading
from tqdm import tqdm, trange
import random
import datasets
import string
import shutil
import os

thread_num = 128
LLVM_BIN_DIR = "llvm17-x86-release/bin/"
OPT_OPTIONS = "-O3 -ffast-math -mavx2 -march=haswell -fno-inline -g"
OFF_OPT_OPTIONS = OPT_OPTIONS+" -Xclang -disable-llvm-passes"
TOOL_BIN = "Tool/build/bin/TOOL"
need_extra_info = True

class RunThread(threading.Thread):
    def __init__(self, thread_name, full_paths, args, outdir, cmd_format, bin_path, output_file_type):
        super(RunThread, self).__init__()
        self.thread_name = thread_name
        self.full_paths = full_paths
        self.args = args
        self.outdir = outdir
        self.cmd_format = cmd_format
        self.bin = bin_path
        self.output_file_type = output_file_type
        self.need_appoint_output_path = True

    def compile(self):
        if len(self.full_paths) == 1:
            for idx, full_path in enumerate(self.full_paths):
                if self.need_appoint_output_path:
                    outputfile = f"{self.outdir}/{shell.getFileNameFromFullFilePath(full_path)}.{self.output_file_type}"
                else:
                    outputfile = self.outdir
                
                cmd = self.cmd_format.format(BIN = self.bin,
                                                        ARGS=self.args,
                                                        InputFile=full_path,
                                                        OutputFile=f"{self.outdir}/{shell.getFileNameFromFullFilePath(full_path)}.{self.output_file_type}")
                # print(cmd)
                res = shell.call_program(cmd)
                print(f"{full_path} over! {self.thread_name}")
            # print(f"{self.thread_name} OVER!")
        else:
            for full_path in tqdm(self.full_paths, desc=self.thread_name):
                if "dummy_func" == shell.getFileNameFromFullFilePath(full_path):
                    continue
                if "synth_exe_wrapper" == shell.getFileNameFromFullFilePath(full_path):
                    continue
                if self.need_appoint_output_path:
                    outputfile = f"{self.outdir}/{shell.getFileNameFromFullFilePath(full_path)}.{self.output_file_type}"
                else:
                    outputfile = self.outdir
                cmd = self.cmd_format.format(BIN = self.bin,
                                                        ARGS=self.args,
                                                        InputFile=full_path,
                                                        OutputFile=outputfile)
                res = shell.call_program(cmd)

    def run(self):
        self.compile()

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def convert_source_code_to_unopt_bc(file_paths, thread_num, outdir):
    threads = []
    i = 0
    for x in batch(file_paths, len(file_paths)//thread_num):
        options = OFF_OPT_OPTIONS
        t = RunThread(thread_name = "Thread"+str(i), 
                          full_paths = x, 
                          args = options, 
                          outdir = outdir,
                          cmd_format = "{BIN} {ARGS} {InputFile} -emit-llvm -c -o {OutputFile}",
                          bin_path = f"{LLVM_BIN_DIR}/clang",
                          output_file_type = "bc")
        t.start()
        i += 1
        threads.append(t)
    
    for t in threads:
        t.join()

# input bc output ll
def normalize(file_paths, thread_num, outdir):
    threads = []
    i = 0
    for x in batch(file_paths, len(file_paths)//thread_num):
        t = RunThread(thread_name = "Thread"+str(i), 
                          full_paths = x, 
                          args = "-discard-value-names", 
                          outdir = outdir,
                          cmd_format = "{BIN} {ARGS} {InputFile} -S -o {OutputFile}",
                          bin_path = f"{LLVM_BIN_DIR}/opt",
                          output_file_type = "ll")
        t.start()
        i += 1
        threads.append(t)
    
    for t in threads:
        t.join()

# def get func-leval opt pass
def get_opt_pass_on_func():
    return "function<eager-inv>(simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;no-switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-sink-common-insts;speculate-blocks;simplify-cond-branch>,sroa<modify-cfg>,early-cse<>,callsite-splitting),function<eager-inv>(mem2reg,instcombine<max-iterations=1000;no-use-loop-info>,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-sink-common-insts;speculate-blocks;simplify-cond-branch>),require<globals-aa>,function(invalidate<aa>),require<profile-summary>,cgscc(devirt<4>(function<eager-inv;no-rerun>(sroa<modify-cfg>,early-cse<memssa>,speculative-execution,jump-threading,correlated-propagation,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-sink-common-insts;speculate-blocks;simplify-cond-branch>,instcombine<max-iterations=1000;no-use-loop-info>,aggressive-instcombine,constraint-elimination,libcalls-shrinkwrap,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-sink-common-insts;speculate-blocks;simplify-cond-branch>,reassociate,loop-mssa(loop-instsimplify,loop-simplifycfg,licm<no-allowspeculation>,loop-rotate<header-duplication;no-prepare-for-lto>,licm<allowspeculation>,simple-loop-unswitch<nontrivial;trivial>),simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-sink-common-insts;speculate-blocks;simplify-cond-branch>,instcombine<max-iterations=1000;no-use-loop-info>,loop(loop-idiom,indvars,loop-deletion,loop-unroll-full),sroa<modify-cfg>,vector-combine,mldst-motion<no-split-footer-bb>,gvn<>,sccp,bdce,instcombine<max-iterations=1000;no-use-loop-info>,jump-threading,correlated-propagation,adce,memcpyopt,dse,move-auto-init,loop-mssa(licm<allowspeculation>),coro-elide,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;hoist-common-insts;sink-common-insts;speculate-blocks;simplify-cond-branch>,instcombine<max-iterations=1000;no-use-loop-info>),function-attrs,function(require<should-not-run-function-passes>))),recompute-globalsaa,function<eager-inv>(float2int,lower-constant-intrinsics,chr,loop(loop-rotate<header-duplication;no-prepare-for-lto>,loop-deletion),loop-distribute,inject-tli-mappings,loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only;>,loop-load-elim,instcombine<max-iterations=1000;no-use-loop-info>,simplifycfg<bonus-inst-threshold=1;forward-switch-cond;switch-range-to-icmp;switch-to-lookup;no-keep-loops;hoist-common-insts;sink-common-insts;speculate-blocks;simplify-cond-branch>,slp-vectorizer,vector-combine,instcombine<max-iterations=1000;no-use-loop-info>,loop-unroll<O3>,transform-warning,sroa<preserve-cfg>,instcombine<max-iterations=1000;no-use-loop-info>,loop-mssa(licm<allowspeculation>),alignment-from-assumptions,loop-sink,instsimplify,div-rem-pairs,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-sink-common-insts;speculate-blocks;simplify-cond-branch>),verify"

def opt(file_paths, thread_num, outdir, logdir):
    threads = []
    i = 0
    OPT_OPTIONS = get_opt_pass_on_func()

    if logdir == None:
        cmd = "{BIN} --passes='{ARGS}' {InputFile} -o {OutputFile}"
    else:
        cmd = "{BIN} --passes='{ARGS}' {InputFile} -o {OutputFile} -print-changed -only-get-changed-pass-name -only-get-changed-pass-name-out-dir="+logdir

    for x in batch(file_paths, len(file_paths)//thread_num):
        t = RunThread(thread_name = "Thread"+str(i), 
                          full_paths = x, 
                          args = OPT_OPTIONS, 
                          outdir = outdir,
                          cmd_format = cmd,
                          bin_path = f"{LLVM_BIN_DIR}/opt",
                          output_file_type = "bc")
        t.start()
        i += 1
        threads.append(t)
    
    for t in threads:
        t.join()

# input ll output bc
def llvm_as(file_paths, thread_num, outdir):
    threads = []
    i = 0
    for x in batch(file_paths, len(file_paths)//thread_num):
        t = RunThread(thread_name = "Thread"+str(i), 
                          full_paths = x, 
                          args = "", 
                          outdir = outdir,
                          cmd_format = "{BIN} {ARGS} {InputFile} -o {OutputFile}",
                          bin_path = f"{LLVM_BIN_DIR}/llvm-as",
                          output_file_type = "bc")
        t.start()
        i += 1
        threads.append(t)
    
    for t in threads:
        t.join()

def delete_and_mkdir(dir_str):
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)
    else:
        shutil.rmtree(dir_str)
        os.makedirs(dir_str)

def read_file_as_str(file_path):
    with open(file_path) as f:
        ctx = f.read()
    return ctx

def write(outpath, lines):
    with open(outpath, "w") as f:
        for line in lines:
            json.dump(line, f)
            f.write("\n")

# collect the unopt ir and opt ir to a whole file
def collect(unopt_files, opt_files, outdir, thread_name):
    file_name_2_unopt_file = {}
    file_name_2_opt_file = {}
    for file_path in unopt_files:
        file_name = shell.getFileNameFromFullFilePath(file_path)
        assert file_name not in file_name_2_unopt_file, f"The `{file_name}` has occured before!\n"
        file_name_2_unopt_file[file_name] = file_path
    for file_path in opt_files:
        file_name = shell.getFileNameFromFullFilePath(file_path)
        assert file_name not in file_name_2_opt_file, f"The `{file_name}` has occured before!\n"
        file_name_2_opt_file[file_name] = file_path
    
    total_len = len(file_name_2_opt_file)
    interval = total_len//thread_name
    records = []
    file_idx = 0
    for file_name in file_name_2_opt_file:
        assert file_name in file_name_2_unopt_file, f"The `{file_name}` must both in opt_dir and unopt_dir!\n"
        unopt_ir = read_file_as_str(file_name_2_unopt_file[file_name])
        opt_ir = read_file_as_str(file_name_2_opt_file[file_name])
        record = {"before_opt_module_ir": unopt_ir, 
                "after_opt_module_ir": opt_ir}
        records.append(record)
        if len(records) == interval:
            write(f"{outdir}/tmp_module_ir_{file_idx}.json", records)
            records.clear()
            file_idx += 1
    if len(records) != 0:
        write(f"{outdir}/tmp_module_ir_{file_idx}.json", records)
        records.clear()
        file_idx += 1

# gen train data info with smaller bb for prompting
def convert_to_smaller_bb_train_data_info(file_paths, thread_num, outdir, TOOL_BIN, inst_threshold):
    threads = []
    i = 0
    for x in batch(file_paths, len(file_paths)//thread_num):
        t = RunThread(thread_name = "Thread"+str(i),
                      full_paths = x,
                      args = "",
                      outdir = outdir,
                      cmd_format = "{BIN} {ARGS} -gen-train-data-with-smaller-bb -input-file-path={InputFile} -output-file-path={OutputFile} -inst-theshold-in-BB="+str(inst_threshold),
                      bin_path = TOOL_BIN,
                      output_file_type = "json")
        t.start()
        i += 1
        threads.append(t)
    
    for t in threads:
        t.join()


def find_last_non_punctuation_char(input_str):
    punctuation = string.punctuation

    # reverse input_str
    reversed_str = input_str[::-1]

    for idx, char in enumerate(reversed_str):
        if char not in punctuation:
            return len(input_str)-(idx+1), char
    return None, None

# gen dataset
def simplify_ir(ir):
    ir_lines = ir.split("\n")
    new_lines = []
    for line in ir_lines:
        new_line = ""
        if line.startswith(";"):
            continue
        tokens = line.split()
        for t in tokens:
            if t == ";":
                break
            if t.startswith("!"):
                break
            if t.startswith("#"):
                continue
            if t.startswith("%struct."):
                last_non_punctuation_index, _ = find_last_non_punctuation_char(t)
                endstr = t[last_non_punctuation_index+1:]
                t = t[:last_non_punctuation_index+1]
                if t.split(".")[-1].isdigit():
                    t = ".".join(t.split(".")[:-1]) + endstr
            new_line += t+" "
        new_line = new_line[:-1]
        if new_line.startswith("define ") and not new_line.endswith("{"):
            new_line += " {"
        if new_line.endswith(","):
            new_line = new_line[:-1]
        new_lines.append(new_line)
    return "\n".join(new_lines)

def get_prompt_format(prompt_kind):
    if prompt_kind == "Unopt2Opt":
        prompt_format = "### Instruction: optimize the following IR with O3\n### LLVM IR:\n```ll\n{LLVMIR}\n```\n{Attribute_List}### Assistant:\n```ll\n{ANS}```\n"
    elif prompt_kind == "CodeRefine":
        prompt_format = "### Instruction: get the target BB O3 optimized IR\n### LLVM IR:\n```ll\n{LLVMIR}\n```\n{Attribute_List}### Assistant for {Target_BB}:\n```ll\n{ANS}```\n### Current Instr to origin:\n{CUR2ORIGIN}\n"
    return prompt_format

def get_opt_BB(BBName2BB):
    BBName2BB = {x.split("label ")[1]: BBName2BB[x] for x in BBName2BB}
    BBWithIRPairs = [(x, simplify_ir(BBName2BB[x])) for x in BBName2BB]
    return BBWithIRPairs

def read_file_as_str(source_file_path):
    ctx = ""
    with open(source_file_path) as f:
        ctx = f.read()
    return ctx

def get_line_num_to_line_map(lines):
    line_num_to_line_map = {}

    cnt = 0
    for line in lines:
        t = line.lstrip()
        if t.startswith("# ") and t.split()[1].isdigit():
            cnt = int(t.split()[1]) - 1
        else:
            line_num_to_line_map[cnt] = line
            cnt += 1
    return line_num_to_line_map

def get_source_code_from_locs(source_file_path, locs):
    ans = set()
    file_ctx = read_file_as_str(source_file_path)
    if file_ctx == "":
        return ""
    file_ctx = file_ctx.split("\n")

    line_num_2_line = get_line_num_to_line_map(file_ctx)
    for line, col in locs:
        # if line != 0:
        #     ans.add(" ".join(line_num_2_line[line-1].split()))
        try:
            if line != 0:
                ans.add(" ".join(line_num_2_line[line-1].split()))
        except:
            raise

    return "\n".join(ans)
    

def format_locs_from_loc_str(loc_str):
    pattern = r'<(\d+),(\d+)>'
    matches = re.findall(pattern, loc_str)
    tuples = [tuple(map(int, match)) for match in matches]
    return tuples

def get_bb_in_with_src(source_file_path, bb_in_str):
    bb_ins = bb_in_str.split("\n")
    res = []
    for bb_in in bb_ins:
        if not bb_in.startswith("inst in label "):
            res.append(bb_in)
            continue
        pattern = r'<(\d+),(\d+)>'
        matches = re.findall(pattern, bb_in)
        tuples = [tuple(map(int, match)) for match in matches]
        if len(tuples) == 0:
            res.append(simplify_ir(bb_in))
        else:
            try:
                src = get_source_code_from_locs(source_file_path, [tuples[-1]])
            except:
                src = "Info loss"
            res.append(simplify_ir(bb_in)+"\n"+src)
    return "\n".join(res)

def gen_prompts(prompt_kind, file_path, InputBeforeIRColumn, OutputAfterIRColumn, ATTRColumns=None):
    PROMPT_FORMAT = get_prompt_format(prompt_kind)
    ATTR_PROMPT_FORMAT = "### {ATTR_KIND}:\n```ll\n{ATTR_CTX}\n```\n"
    prompts = []
    with open(file_path) as f:
        print(file_path)
        for line in f.readlines():
            record = json.loads(line)
            
            if simplify_ir(record['beforeFuncIR']) == simplify_ir(record['originAfterFuncIR']): #delete unchanged
                    continue
            
            if ATTRColumns == None:
                ATTRColumns = set(record.keys()) - {InputBeforeIRColumn, OutputAfterIRColumn}
            attributes = ""
            for attr in ATTRColumns:
                if attr not in record:
                    continue
                attributes += ATTR_PROMPT_FORMAT.format(ATTR_KIND=attr,
                                                        ATTR_CTX=record[attr])+"\n"
            
            if prompt_kind == "Unopt2Opt":
                prompt = PROMPT_FORMAT.format(LLVMIR=simplify_ir(record[InputBeforeIRColumn]),
                                            ANS=simplify_ir(record[OutputAfterIRColumn]))
                prompts.append(prompt)
            elif prompt_kind == "CodeRefine":
                BBWithIRPairs = get_opt_BB(record['afterBBName2BB'])
                BB2BBIn = record['afterBBName2BBIn']
                tmpAfterBB2BBOut = record['afterBBName2BBOut']
                afterBB2BBOut = {}
                BB2BBIn = {x.split("label ")[1]: get_bb_in_with_src(record['sourceFile'], BB2BBIn[x]) for x in BB2BBIn}
                tmpBB2LastSSA = record['afterBBName2StartWith']
                BB2LastSSA = {}
                for x in tmpBB2LastSSA:
                    if tmpBB2LastSSA[x] == "":
                        dest = ""
                    else:
                        dest = tmpBB2LastSSA[x].split()[-1]
                        assert dest.startswith("%"), "ERROR: the last SSA in BB not startwith %!\n"
                    BB2LastSSA[x.split("label ")[1]] = dest

                BB2OriginI = record['afterBBName2OriginI']
                BB2OriginI = {x.split("label ")[1]:simplify_ir(BB2OriginI[x]) for x in BB2OriginI}
                
                for x in tmpAfterBB2BBOut:
                    intrs = tmpAfterBB2BBOut[x].split(";")
                    dest = ""
                    for inst in intrs:
                        if inst == "":
                            continue
                        dest += inst.split()[-1] + ","
                        # print(dest)
                        assert inst.split()[-1].startswith("%"), "ERROR: the out in BB not startwith %!\n"
                    afterBB2BBOut[x.split("label ")[1]] = dest
                for bb, bbir in BBWithIRPairs:
                    # global need_extra_info
                    if need_extra_info:
                        full_attr = attributes + ATTR_PROMPT_FORMAT.format(ATTR_KIND=f"{bb} IN",
                                        ATTR_CTX=BB2BBIn[bb])
                    else:
                        full_attr = attributes
    
                    full_attr = full_attr + f"### The Last SSA: {BB2LastSSA[bb]}\n"
                    if need_extra_info:
                        prompt = PROMPT_FORMAT.format(LLVMIR=simplify_ir(record[InputBeforeIRColumn]),
                                                Attribute_List=full_attr,
                                                Target_BB=bb,
                                                ANS=bbir,
                                                CUR2ORIGIN=BB2OriginI[bb])
                    else:
                        prompt = PROMPT_FORMAT.format(LLVMIR=simplify_ir(record[InputBeforeIRColumn]),
                                                Attribute_List=full_attr,
                                                Target_BB=bb,
                                                ANS=bbir,
                                                CUR2ORIGIN="None")
                    prompts.append(prompt)
    return prompts
            
def merge_log(file_paths, log_dir, output_dir):
    log_files = shell.getAllFilesFromDir(log_dir)
    log_files = [x.replace("//", "/") for x in log_files]
    for file_path in file_paths:
        file_name = shell.getFileNameFromFullFilePath(file_path)
        records = []
        with open(file_path) as f:
            print(file_path)
            for line in f.readlines():
                record = json.loads(line)
                log_file_name = record["sourceFileWithFuncName"]
                log_file_path = f"{log_dir}/{log_file_name}".replace("//", "/")
                if log_file_path not in log_files:
                    record["ChangedPass"] = ""
                else:
                    record["ChangedPass"] = ",".join(set(read_file_as_str(log_file_path).split(",")))
                records.append(record)
        with open(f"{output_dir}/{file_name}.json", "w") as wf:
            for r in records:
                json.dump(r, wf)
                wf.write("\n")

def save_dataset(prompts, output_dir):
    random.shuffle(prompts)
    dictt = {"text": prompts}
    dataset = datasets.arrow_dataset.Dataset.from_dict(dictt)
    dataset.save_to_disk(output_dir)

def get_all_files(input_dir):
    file_paths = shell.getAllFilesFromDir(input_dir)
    assert len(file_paths) != 0, f"There is no file in `{input_dir}`!\n"
    return file_paths

def collect_stage_to_opt_module_ir(input_dir, output_dir):
    delete_and_mkdir(f"{output_dir}/O3_bc")
    delete_and_mkdir(f"{output_dir}/log")
    global thread_name
    opt(get_all_files(input_dir), 
        thread_num, 
        f"{output_dir}/O3_bc", 
        f"{output_dir}/log")
    
    delete_and_mkdir(f"{output_dir}/O3_normalize")
    normalize(get_all_files(f"{output_dir}/O3_bc"), 
              thread_num, 
              f"{output_dir}/O3_normalize")

    delete_and_mkdir(f"{output_dir}/tmp_module_ir")
    collect(get_all_files(f"{input_dir}"),
            get_all_files(f"{output_dir}/O3_normalize"), 
            f"{output_dir}/tmp_module_ir", thread_num)    

parser = argparse.ArgumentParser(description='Gen Dataset')
parser.add_argument('-m','--mode')
parser.add_argument('-i','--input_dir')
parser.add_argument('-unopt-dir','--unopt_dir')
parser.add_argument('-opt-dir','--opt_dir')
parser.add_argument('-o','--output_dir')
parser.add_argument('-log-dir', '--log_dir')
parser.add_argument('-prompt-kind','--prompt_kind')
parser.add_argument('-input-column-name','--input_column')
parser.add_argument('-output-column-name','--output_column')
parser.add_argument('-attribute-column-names', '--attribute_columns')
parser.add_argument('-inst-threshold','--inst_threshold')
parser.add_argument('-need-extra-info', '--need_extra_info')

args = parser.parse_args()

if len(get_all_files(args.input_dir)) < thread_num:
    thread_num = len(get_all_files(args.input_dir))
delete_and_mkdir(f"{args.output_dir}/raw_input_bc")
convert_source_code_to_unopt_bc(get_all_files(args.input_dir), 
                                thread_num,
                                f"{args.output_dir}/raw_input_bc")

delete_and_mkdir(f"{args.output_dir}/input_normalize")
normalize(get_all_files(f"{args.output_dir}/raw_input_bc"), 
            thread_num, 
            f"{args.output_dir}/input_normalize")

collect_stage_to_opt_module_ir(f"{args.output_dir}/input_normalize", f"{args.output_dir}")
delete_and_mkdir(f"{args.output_dir}/smaller_bb_info")
convert_to_smaller_bb_train_data_info(get_all_files(f"{args.output_dir}/tmp_module_ir"), 
                                            thread_num,
                                            f"{args.output_dir}/smaller_bb_info", TOOL_BIN, 10)

delete_and_mkdir(f"{args.output_dir}/smaller_bb_info_log")
merge_log(get_all_files(f"{args.output_dir}/smaller_bb_info"), 
            f"{args.output_dir}/log", 
            f"{args.output_dir}/smaller_bb_info_log")

delete_and_mkdir(f"{args.output_dir}/dataset")
file_paths = get_all_files(f"{args.output_dir}/smaller_bb_info_log")
prompts = []
for file_path in file_paths:
    tprompts = gen_prompts("CodeRefine",
                            file_path, 
                            "beforeFuncIR", 
                            None,
                            ["beforeFuncLoopInfo","ChangedPass","afterCFGWithSmallerBB"])
    prompts.extend(tprompts)
save_dataset(prompts, f"{args.output_dir}/dataset/CodeRefine")

prompts = []
for file_path in file_paths:
    tprompts = gen_prompts("Unopt2Opt",
                            file_path, 
                            "beforeFuncIR", 
                            "originAfterFuncIR")
    prompts.extend(tprompts)
save_dataset(prompts, f"{args.output_dir}/dataset/Unopt2Opt")