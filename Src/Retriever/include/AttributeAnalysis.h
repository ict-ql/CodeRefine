#ifndef __ATTRIBUTE_ANALYSIS__
#define __ATTRIBUTE_ANALYSIS__

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
// #include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/CallGraph.h"

using namespace llvm;


class AttributeAnalysisPass : public PassInfoMixin<AttributeAnalysisPass> {
public:
    std::list<std::string> rootFuncNames;
    std::list<std::string> basicAPIs;
    AttributeAnalysisPass(std::list<std::string> rootFuncNames,
                            std::list<std::string> basicAPIs) 
                            : rootFuncNames(rootFuncNames), basicAPIs(basicAPIs) {}

    PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

void runAttributeAnalysis(std::string InputFilePath, std::string FocusFuncFilePath, 
                            std::string BasicAPIFilePath, std::string OutputFilePath);
#endif