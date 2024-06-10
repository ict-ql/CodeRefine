#include "GenFuncTrainData.h"
#include <iostream>
#include <fstream>
#include <memory>
#include "json/json.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/MemoryBuffer.h>
#include "llvm/IR/LegacyPassManager.h"
#include "utils.h"

#include "CollectModuleAnalysisInfo.h"

using namespace std;
using namespace llvm;

string getFuncIR(unique_ptr<Module> &M, 
                        string funcName){
    string funcIRStr = "";
    for (Function &F: *M){
        if(F.getName() == funcName){
            // break;
            raw_string_ostream OS(funcIRStr);
            F.print(OS);
            return funcIRStr;
        }
    }
    assert(0 && "ERROR: The Module IR does not contain changed func!\n");
    return funcIRStr;
}

map<string, string> getFuncIRWithCFGAndDF(string module_ir_str){
    map<string, string> infos;
    unique_ptr<Module> M;
    try{
        M = getLLVMIR(module_ir_str);
    } catch(const char* msg){
        // errs()<<"WARNING: the module ir cannot be parsed!\n";
        errs()<<msg<<"\n";
        infos["ir"] = "";
        return infos;
    }
    string funcIRStr = "";
    Function *tmpF = NULL;
    for (Function &F: *M){
        if(F.isDeclaration())
            continue;
        assert(tmpF == NULL && "The module_ir_str has more than one define func!\n");
        tmpF = &F;
    }
    if(tmpF == NULL){
        infos["ir"] = "";
        return infos;
    }

    //rename the first BB's name 'B_entry'
    tmpF->front().setName("B_entry");

    raw_string_ostream OS(funcIRStr);
    tmpF->print(OS);
    infos["ir"] = funcIRStr;
    infos["source_filename"] = M->getSourceFileName();
    ModulePassManager PM;
    ModuleAnalysisManager AM;
    PM.addPass(createModuleToFunctionPassAdaptor(CollectModuleAnalysisInfoPass()));
    PM.run(*M, AM);
    
    assert(CollectModuleAnalysisInfoPass::Func2AnalysisInfo.count(tmpF) > 0 && "The Function must have analysis info!\n");
    // for(auto KV: CollectModuleAnalysisInfoLegacy::Func2AnalysisInfo[tmpF]){
    //     infos[KV.first] = KV.second;
    // }
    infos["cfg_info"] = CollectModuleAnalysisInfoPass::Func2AnalysisInfo[tmpF].CFGRelatedResult;
    infos["loop_info"] = CollectModuleAnalysisInfoPass::Func2AnalysisInfo[tmpF].LoopInfoResult;

    CollectModuleAnalysisInfoPass::Func2AnalysisInfo.clear();
    return infos;
}

void printTrainDataToOutputFile(string &beforeFuncIR, 
                                string &afterFuncIR,
                                ofstream &outfile,
                                Json::StreamWriter* writer){
    Json::Value root;
    root["before_ir"] = beforeFuncIR;
    root["after_ir"] = afterFuncIR;
    writer->write(root, &outfile);
    outfile<<"\n";
}

void printTrainDataToOutputFile(string &funcName,
                                string &passInfo,
                                string &FuncIR,
                                ofstream &outfile,
                                Json::StreamWriter* writer){
    Json::Value root;
    root["func_name"] = funcName;
    root["pass_info"] = passInfo;
    root["ir"]        = FuncIR;
    writer->write(root, &outfile);
    outfile<<"\n";
}

void printEachFuncToOutputFile(unique_ptr<Module> &M,
                                ofstream &outfile,
                                Json::StreamWriter* writer){
    Json::Value root;
    for (Function &F: *M){
        if(F.isDeclaration())
            continue;
        string func_name = F.getName().str();

        string funcIRStr = "";
        raw_string_ostream OS(funcIRStr);
        F.print(OS);
        root[func_name] = funcIRStr;
    }
    writer->write(root, &outfile);
    outfile<<"\n";
}

void genFunctionTrainData(string InputFilePath, string OutputFilePath){
    Json::Reader reader;
    Json::Value root;
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    Json::StreamWriter* writer = builder.newStreamWriter();

    ofstream outfile;
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");

    // parse json file
    ifstream infile(InputFilePath);
    assert(infile.is_open() && "ERROR: open input file failed!\n");

    string line;

    // create opt pipeline    
    while(getline(infile, line)){
        if(reader.parse(line, root)){
            string opt_ir = root["opt_ir"].asString();
            string no_opt_ir = root["no_opt_ir"].asString();
            string mem2reg_ir = root["mem2reg_ir"].asString();
            string asm_txt = root["asm"].asString();
            string mir = root["mir"].asString();
            string llc_asm = root["llc_asm"].asString();
            string source_code = root["source_code"].asString();

            map<string, string> opt_func_ir_infos = getFuncIRWithCFGAndDF(opt_ir);
            map<string, string> no_opt_func_ir_infos = getFuncIRWithCFGAndDF(no_opt_ir);
            map<string, string> mem2reg_func_ir_infos = getFuncIRWithCFGAndDF(mem2reg_ir);

            string opt_func_ir = opt_func_ir_infos["ir"];
            string no_opt_func_ir = no_opt_func_ir_infos["ir"];
            string mem2reg_func_ir = mem2reg_func_ir_infos["ir"];
            
            if(opt_func_ir == "" || no_opt_func_ir == "" || mem2reg_func_ir == "")
                continue;

            Json::Value root;
            root["source_filename"] = mem2reg_func_ir_infos["source_filename"];
            root["opt_ir"] = opt_func_ir;
            root["no_opt_func_ir"] = no_opt_func_ir;
            root["mem2reg_func_ir"] = mem2reg_func_ir;
            root["mem2reg_cfg_info"] = mem2reg_func_ir_infos["cfg_info"];
            root["mem2reg_loop_info"] = mem2reg_func_ir_infos["loop_info"];
            root["asm"] = asm_txt;
            root["mir"] = mir;
            root["llc_asm"] = llc_asm;
            root["source_code"] = source_code;

            writer->write(root, &outfile);
            outfile<<"\n";
            // break;
        }
    }
    infile.close();

    outfile.close();
}

void genFunctionTrainData(vector<string> InputFilePaths, string OutputFilePath){
    Json::Reader reader;
    Json::Value root;
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    Json::StreamWriter* writer = builder.newStreamWriter();

    ofstream outfile;
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");

    for(string Strr: InputFilePaths){
        // parse json file
        ifstream infile(Strr);
        assert(infile.is_open() && "ERROR: open input file failed!\n");

        string line;
        while(getline(infile, line)){
            if(reader.parse(line, root)){
                string funcName = root["func_name"].asString();
                string beforeIR = root["before_ir"].asString();
                string afterIR = root["after_ir"].asString();

                unique_ptr<Module> beforeMIR = getLLVMIR(beforeIR);
                string beforeFuncIR = getFuncIR(beforeMIR, funcName);
                unique_ptr<Module> afterMIR = getLLVMIR(afterIR);
                string afterFuncIR = getFuncIR(afterMIR, funcName);
                
                printTrainDataToOutputFile(beforeFuncIR, afterFuncIR, outfile, writer);
            }
        }
        infile.close();
    }
    outfile.close();
}

/*把切分pass的log文件的ModuleIR转成FunctionIR*/
void processSplitPassData(vector<string> InputFilePaths, string OutputFilePath){
    Json::Reader reader;
    Json::Value root;
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    Json::StreamWriter* writer = builder.newStreamWriter();

    ofstream outfile;
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");

    for(string Strr: InputFilePaths){
        // parse json file
        ifstream infile(Strr);
        assert(infile.is_open() && "ERROR: open input file failed!\n");

        string line;
        while(getline(infile, line)){
            if(reader.parse(line, root)){
                string funcName = root["func_name"].asString();
                string passInfo = root["pass_info"].asString();
                string IR = root["ir"].asString();

                unique_ptr<Module> MIR = getLLVMIR(IR);
                string FuncIR = getFuncIR(MIR, funcName);

                printTrainDataToOutputFile(funcName, passInfo, FuncIR, outfile, writer);
            }
        }
        infile.close();
    }
    outfile.close();
}

//输入ModuleIR.ll输出json文件，key是function的名字，value是functionIR
void genJustFuncIRFromModuleIR(string InputFilePath, string OutputFilePath){
    Json::Reader reader;
    Json::Value root;
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    Json::StreamWriter* writer = builder.newStreamWriter();

    ofstream outfile;
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");

    auto MIR = getLLVMIRFromFile(InputFilePath);
    printEachFuncToOutputFile(MIR, outfile, writer);
    outfile.close();
}