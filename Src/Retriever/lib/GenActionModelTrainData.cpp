#include "GenActionModelTrainData.h"
#include "CollectModuleAnalysisInfo.h"

#include "utils.h"

using namespace std;
using namespace llvm;

/**
 * @brief convert the map that key is Function to a new map whose keys is Function Name
*/
map<string, AnalysisInfo> convertKeyToFunctionName(map<Function*, AnalysisInfo> &Mapp){
    map<string, AnalysisInfo> res;
    for(auto &KV: Mapp){
        res[KV.first->getName().str()] = KV.second;
    }
    return res;
}

/**
 * @brief convert info to records in json
*/
vector<map<string, string>> 
    getRecordsFromAnalysisInfo(map<Function*, AnalysisInfo> &beforeAnalysisInfo,
                                map<Function*, AnalysisInfo> &afterAnalysisInfo){
    vector<map<string, string>> records;

    // the func name in one module ir cannot be reduplicative
    map<string, AnalysisInfo> bFuncName2AnalysisInfo = convertKeyToFunctionName(beforeAnalysisInfo);
    map<string, AnalysisInfo> aFuncName2AnalysisInfo = convertKeyToFunctionName(afterAnalysisInfo);

    for(auto &KV: aFuncName2AnalysisInfo){
        string funcName = KV.first;
        if(bFuncName2AnalysisInfo.find(funcName) == bFuncName2AnalysisInfo.end()){
            errs()<<funcName<<"\n";
            records.clear(); // there is a module pass that we cannot deal with
            break;
            // continue; // the function occures because of module optimize(for example, constant propogation)
        }
        // assert(bFuncName2AnalysisInfo.find(funcName) != bFuncName2AnalysisInfo.end() && "the funcName in module ir after optimized must occure in odule ir before optimized!\n");
        AnalysisInfo bAI = bFuncName2AnalysisInfo[funcName];
        AnalysisInfo aAI = aFuncName2AnalysisInfo[funcName];
        map<string, string> bAnalysisInfo= bAI.getNoEmptyFields();
        map<string, string> aAnalysisInfo= aAI.getNoEmptyFields();

        map<string, string> record;
        for(auto bKV: bAnalysisInfo){
            record["before"+bKV.first] = bKV.second;
        }
        record["afterFuncIR"] = aAnalysisInfo["FuncIR"];
        records.push_back(record);
    }
    return records;
}

void genRawTrainDataForPrompt(string InputFilePath, string OutputFilePath){
    Json::Reader reader;
    Json::Value root;
    ifstream infile(InputFilePath);
    assert(infile.is_open() && "ERROR: open input file failed!\n");
    ofstream outfile;
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");

    string line;
    while(getline(infile, line)){
        if(reader.parse(line, root)){
            string beforeModuleIRstr = root["before_opt_module_ir"].asString();
            string afterModuleIRstr = root["after_opt_module_ir"].asString();
            unique_ptr<Module> beforeModuleIR;
            unique_ptr<Module> afterModuleIR;
            try{
                beforeModuleIR = getLLVMIR(beforeModuleIRstr);
                afterModuleIR = getLLVMIR(afterModuleIRstr);
            }catch(const char* msg){
                errs()<<msg<<"\n";
                continue;
            }
            assert(beforeModuleIR.get() && afterModuleIR.get() && "ERROR: must not empty");
            map<Function*, AnalysisInfo> beforeAnalysisInfo = runToCollectModuleAnalysisInfo(beforeModuleIR);
            map<Function*, AnalysisInfo> afterAnalysisInfo = runToCollectModuleAnalysisInfo(afterModuleIR);
            vector<map<string, string>> records = getRecordsFromAnalysisInfo(beforeAnalysisInfo,
                                                                                afterAnalysisInfo);
            writeRecordsIntoJson(records, outfile);
        }
    }

    infile.close();
    outfile.close();
}