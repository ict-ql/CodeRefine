#include "utils.h"
#include "GenAnalysisTrainData.h"
#include "AnalysisInfo.h"
#include "CollectModuleAnalysisInfo.h"

#include <iostream>
#include <fstream>
#include <memory>
#include "json/json.h"
#include <map>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"

using namespace std;
using namespace llvm;

string FuncIRResFileName="Data_FuncIR.json";
string SpecialTokenFileName="SpecialToken";

#include <set>
void get_opcode_and_data_type_in_Module(unique_ptr<Module> &MIR, set<string> &special_tokens){
    string tmp = "";
    raw_string_ostream OS(tmp);
    for(auto &F: *MIR){
        for(auto &BB: F){
            for(auto &I: BB){
                special_tokens.insert(I.getOpcodeName());
                I.getType()->print(OS);
                special_tokens.insert(tmp);
                tmp = "";
            }
        }
    }
}

void writeSpecialTokens(set<string> &special_tokens, ofstream &out){
    for(string strr: special_tokens){
        out<<strr+"\n";
    }
}

void genAnalysisInfoTrainData(vector<string> InputFilePaths, string OutputFileDir){
    // init for write
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    Json::StreamWriter* writer = builder.newStreamWriter();

    // open output file
    string OutputFuncIRFilePath = OutputFileDir+"/"+FuncIRResFileName;
    ofstream outfile_funcir;
    outfile_funcir.open(OutputFuncIRFilePath, std::ios::out | std::ios::trunc);

    string OutputSpecialTokenFilePath = OutputFileDir+"/"+SpecialTokenFileName;
    ofstream outfile_stokens;
    outfile_stokens.open(OutputSpecialTokenFilePath, std::ios::out | std::ios::trunc);
    assert(outfile_funcir.is_open() && outfile_stokens.is_open() && "ERROR: open output file failed!\n");

    // create opt pipeline
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(CollectModuleAnalysisInfoPass()));

    // initial
    Json::Reader reader;
    Json::Value root;

    map<Function*, AnalysisInfo> func2AnalysisInfo;
    set<string> special_tokens;
    for(string inputFilePath: InputFilePaths){
        ifstream infile(inputFilePath);
        assert(infile.is_open() && "ERROR: open input file failed!\n");

        string line;
        while(getline(infile, line)){
            if(reader.parse(line, root)){
                string pass_name = root["pass_name"].asString();
                string before_ir = root["before_ir"].asString();
                string after_ir = root["after_ir"].asString();
                
                // errs()<<ir<<"\n";
                unique_ptr<Module> after_MIR = getLLVMIR(after_ir);
                get_opcode_and_data_type_in_Module(after_MIR, special_tokens);

                // run the pass
                MPM.run(*after_MIR, MAM);

                // @TODO: get the result
                // write with before ir
                // free the result
            }
        }
        infile.close();
    }
    writeSpecialTokens(special_tokens, outfile_stokens);

    outfile_funcir.close();
    outfile_stokens.close();
}