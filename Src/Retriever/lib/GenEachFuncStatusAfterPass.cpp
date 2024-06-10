#include "GenEachFuncStatusAfterPass.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <map>
#include "json/json.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/MemoryBuffer.h>
#include "llvm/IR/LegacyPassManager.h"
#include "utils.h"

using namespace std;
using namespace llvm;

class ModuleIRInfo{
public:
    string functionName; // if the pass is Module Pass, the var can be ''; the function name may be a BB name(loopPass), if we cannot find the function name in ModuleIR, we assume each function in ModuleIR changed after passName.
    string passName; // if dump at start, the var can be ''
    string moduleIR;

    string getStr(){
        string res = "FunctionName: "+functionName+"\t\tPassName: "+passName+"\n"+moduleIR+"\n";
        return res;
    }
};

class FunctionPtrInfo{
public:
    string passName;
    Function *F;
};

/**
 * @brief If the str startswith substr, return true, else false
*/
bool startswith(string str, string substr){
    if (str.rfind(substr, 0) == 0){
        return true;
    }
    return false;
}

bool contains(string str, string substr){
    if (str.find(substr) != std::string::npos){
        return true;
    }
    return false;
}

bool contains(map<string, vector<FunctionPtrInfo>> key2vals, string key){
    if (key2vals.find(key) != key2vals.end())
        return true;
    return false;
}

/**
 * @brief 输出是这个文件的line list，不包括\n
*/
vector<string> readlinesFromFile(string InputFilePath){
    vector<string> lines;

    ifstream infile(InputFilePath);
    assert(infile.is_open() && "ERROR: open input file failed!\n");
    string line;
    while(getline(infile, line)){
        lines.push_back(line);
    }
    return lines;
}

string getPassNameFromBanner(string banner){
    if(banner == "*** IR Dump At Start ***")
        return "";
    string start = "*** IR Dump After ";
    string end = " on ";
    size_t pos1 = banner.find(start);
    size_t pos2 = banner.find(end);
    // string errorMsg = "ERROR: cannot regnize `"+banner+"` banner format!\n";
    assert((pos1 != std::string::npos && pos2 != std::string::npos && pos1 < pos2) && "ERROR: cannot regnize banner format!\n");
    return banner.substr(pos1 + start.size(), pos2 - pos1 - start.size());
}

string getFunctionNameFromBanner(string banner){
    if(banner == "*** IR Dump At Start ***")
        return "";
    if(contains(banner, " [module] ***"))
        return "";
    string start = " on ";
    string end = " ***";
    size_t pos1 = banner.find(start);
    size_t pos2 = banner.find(end);
    // string errorMsg = "ERROR: cannot regnize `"+banner+"` banner format!\n";
    assert((pos1 != std::string::npos && pos2 != std::string::npos && pos1 < pos2) && "ERROR: cannot regnize banner format!\n");
    return banner.substr(pos1 + start.size(), pos2 - pos1 - start.size());
}

/**
 * @brief The function will change the `*ptrI` and add element to `moduleIRInfos` 
*/
void analysisModuleIRInfo(vector<string> lines, 
                            int *ptrI, 
                            vector<ModuleIRInfo> &moduleIRInfos){
    int currentI = *ptrI;
    string banner = lines[currentI];
    string passName = getPassNameFromBanner(lines[currentI]);
    string funcName = getFunctionNameFromBanner(lines[currentI]);

    assert(contains(lines[currentI+1], "; ModuleID = '") && "ERROR: the next line of banner not in ModuleIR format!\n");
    
    // get the module ir
    string moduleIR = "";
    currentI += 1;
    while(currentI < lines.size()){
        if(startswith(lines[currentI], "*** IR Dump ")){
            break;
        }
        if(startswith(lines[currentI], "*** IR Pass ")){
            break;
        }
        moduleIR += lines[currentI]+"\n";
        currentI += 1;
    }

    ModuleIRInfo mirInfo;
    mirInfo.functionName = funcName;
    mirInfo.passName = passName;
    mirInfo.moduleIR = moduleIR;

    *ptrI = currentI;

    moduleIRInfos.push_back(mirInfo);
}

vector<ModuleIRInfo> getModuleIRInfosFromLog(string InputFilePath){
    vector<ModuleIRInfo> moduleIRInfos;

    vector<string> lines = readlinesFromFile(InputFilePath);
    int i = 0;
    while(i < lines.size()){
        if(startswith(lines[i], "*** IR ")){
            if(contains(lines[i], "omitted because no change ***") ||
                contains(lines[i], " ignored ***")){
                i++;
                continue;
            }
            // errs()<<lines[i]<<"\t"<<getPassNameFromBanner(lines[i])
            //                 <<"\t"<<getFunctionNameFromBanner(lines[i])<<"\n";
            analysisModuleIRInfo(lines, &i, moduleIRInfos);
        }   
        else
            i++;
    }
    return moduleIRInfos;
}

/**
 * @brief if functionName == "" or there is no such functionName in Module, return ALL Function,
 * else just return one function.
*/
vector<Function*> getFunctionPtr(unique_ptr<Module> &M, string functionName, bool *containF){
    vector<Function*> funcPtrs;
    *containF = false;
    for (Function &F: *M){
        if(F.isDeclaration())
            continue;
        if(F.getName() == functionName){
            funcPtrs.push_back(&F);
            assert(funcPtrs.size() == 1 && "");
            *containF = true;
            break;
        }
    }
    if(funcPtrs.size() == 0){
        for (Function &F: *M){
            if(F.isDeclaration())
                continue;
            funcPtrs.push_back(&F);
        }
    }
    return funcPtrs;
}

/**
 * @brief FuncInfo 指的是"source code+function name"，能稍微减轻一些重命名的问题
*/
map<string, vector<FunctionPtrInfo>> getFuncInfo2FunctionStatus(map<ModuleIRInfo*, unique_ptr<Module>> &moduleIRInfo2MPtr,
                                                                vector<ModuleIRInfo> &moduleIRInfos){
    map<string, vector<FunctionPtrInfo>> fileNameFunctionName2FunctionPtrInfo;
    for(ModuleIRInfo &moduleIRInfo: moduleIRInfos){
        string mir = moduleIRInfo.moduleIR;
        string passName = moduleIRInfo.passName;
        string functionName = moduleIRInfo.functionName;
        string sourceFileName = moduleIRInfo2MPtr[&moduleIRInfo]->getSourceFileName();
        bool containF = false;
        vector<Function*> funcPtrs = getFunctionPtr(moduleIRInfo2MPtr[&moduleIRInfo], functionName, &containF);
        assert(funcPtrs.size() > 0 && "ERROR: ModuleIR doesn't contain any function!\n");
        if(containF){
            Function* F = funcPtrs[0];
            FunctionPtrInfo FInfo;
            FInfo.F = F;
            FInfo.passName = passName;
            string fileNameFunctionName = sourceFileName+"/"+functionName;
            if(contains(fileNameFunctionName2FunctionPtrInfo,
                        fileNameFunctionName)){
                fileNameFunctionName2FunctionPtrInfo[fileNameFunctionName].push_back(FInfo);
            }else{
                fileNameFunctionName2FunctionPtrInfo[fileNameFunctionName] = {FInfo};
            }
            continue;
        }
        // check: each pass start at "*** IR Dump At Start ***", so at this time `fileNameFunctionName2FunctionPtrInfo` must be empty.
        if(passName == "" && functionName == ""){
            passName = "START";
            for(Function* F: funcPtrs){
                string fileNameFunctionName = sourceFileName+"/"+F->getName().str();
                assert(!contains(fileNameFunctionName2FunctionPtrInfo,
                                fileNameFunctionName) && "ERROR: each pass start at `*** IR Dump At Start ***`, so at this time `fileNameFunctionName2FunctionPtrInfo` must be empty!\n");
                fileNameFunctionName2FunctionPtrInfo[fileNameFunctionName] = {};
            }
        }

        for(Function* F: funcPtrs){
            FunctionPtrInfo FInfo;
            FInfo.F = F;
            FInfo.passName = passName;
            string fileNameFunctionName = sourceFileName+"/"+F->getName().str();
            if(contains(fileNameFunctionName2FunctionPtrInfo,
                        fileNameFunctionName)){
                fileNameFunctionName2FunctionPtrInfo[fileNameFunctionName].push_back(FInfo);
            }else{
                fileNameFunctionName2FunctionPtrInfo[fileNameFunctionName] = {FInfo};
            }
        }
    }
    return fileNameFunctionName2FunctionPtrInfo;
}

string getFuncStr(Function *F){
    string funcIRStr = "";
    raw_string_ostream OS(funcIRStr);
    F->print(OS);
    return funcIRStr;
}

vector<FunctionPtrInfo> getOnlyChangedPass(vector<FunctionPtrInfo> funcInfos){
    vector<FunctionPtrInfo> changedFuncInfos;
    string previousFIR = "";
    for(FunctionPtrInfo fpInfo: funcInfos){
        Function *F = fpInfo.F;
        // F->print(errs());
        string currentFIR = getFuncStr(F);
        if(previousFIR == ""){
            previousFIR = currentFIR;
            changedFuncInfos.push_back(fpInfo);
            continue;
        }
        if(previousFIR != currentFIR){
            previousFIR = currentFIR;
            changedFuncInfos.push_back(fpInfo);
        }
    }
    return changedFuncInfos;
}

void genEachFuncStatusAfterPass(string InputFilePath, string OutputFileDir){
    vector<ModuleIRInfo> moduleIRInfos = getModuleIRInfosFromLog(InputFilePath);

    map<ModuleIRInfo*, unique_ptr<Module>> moduleIRInfo2MPtr;
    for(ModuleIRInfo &moduleIRInfo: moduleIRInfos){
        string mir = moduleIRInfo.moduleIR;
        unique_ptr<Module> M = getLLVMIR(mir);
        moduleIRInfo2MPtr[&moduleIRInfo] = move(M);
    }

    map<string, vector<FunctionPtrInfo>> fileNameFunctionName2FunctionPtrInfo = getFuncInfo2FunctionStatus(moduleIRInfo2MPtr, moduleIRInfos);
    // delete some unchanged pass
    map<string, vector<FunctionPtrInfo>> fileNameFunctionName2ChangedFunctionPtrInfo;
    for(auto &KV: fileNameFunctionName2FunctionPtrInfo){
        fileNameFunctionName2ChangedFunctionPtrInfo[KV.first] = getOnlyChangedPass(KV.second);
    }
    for(auto &KV: fileNameFunctionName2ChangedFunctionPtrInfo){
        errs()<<KV.first<<":\n";
        for(auto fpInfo: KV.second){
            errs()<<"-------------"<<fpInfo.passName<<"-------------\n";
            fpInfo.F->print(errs());
        }
        errs()<<"\n";
    }
    
}

// @TODO: gen Each BB Status