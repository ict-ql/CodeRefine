#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/MemorySSA.h"
#include "AnalysisInfo.h"
#include <fstream>
#include <iostream>
#include <memory>
#include "json/json.h"
#include <map>

using namespace llvm;

namespace llvm{
  FunctionPass *createCollectModuleAnalysisInfoPass();
  void initializeCollectModuleAnalysisInfoLegacyPass(PassRegistry&);
}

void runCollectModuleAnalysisInfoTest(std::string InputFilePath);
std::map<Function*, AnalysisInfo> 
    runToCollectModuleAnalysisInfo(std::unique_ptr<Module> &M);

class CollectModuleAnalysisInfoPass : public PassInfoMixin<CollectModuleAnalysisInfoPass> {
public:
    // static std::map<Function*, std::map<std::string, std::string>> Func2AnalysisInfo;
    static std::map<Function*, AnalysisInfo> Func2AnalysisInfo;
    AnalysisInfo analysisInfoResult;
    
    CollectModuleAnalysisInfoPass(){}

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
    std::string  getFuncIR(Function &F);
    std::string getBlockFrequencyInfo(BlockFrequencyInfo *BFI);
    std::string getDemandedBitsInfo(DemandedBits *DB);
    std::string getDominatorTreeInfo(DominatorTree *DT);
    std::string getLoopAccessInfoManagerInfo(LoopAccessInfoManager *LAIs, LoopInfo *LI);
    std::string getLoopInfo(LoopInfo *LI);
    std::string getMemorySSAInfo(MemorySSA *MSSA);
    std::string getScalarEvolutionInfo(Function &F, ScalarEvolution *SE, LoopInfo *LI);
    std::string getCFGRelatedResultInfo(Function &F, BlockFrequencyInfo *BFI);
    std::string getInstructionAttrResultInfo(Function &F);
    std::string getInstructionUserInfo(Instruction *I);
    std::string getInstructionOperandInfo(Instruction *I);
    std::string getInstructionStr(Instruction *I);
    std::string getDFRelatedResultInfo(Function &F, MemorySSA *MSSA, LoopAccessInfoManager *LAIs, LoopInfo *LI, ScalarEvolution *SE);
    std::string getAliasAnalysisResultInfo(Function &F, AAResults *AA);
    std::string getDataLayoutResultInfo(Function &F);
    std::string getSCEVInfo(ScalarEvolution *SE, Instruction *I);
    std::string getGlobalVarInfo(Function &F);
    std::string getFunctionDeclareInfo(Function &F);
    
    AnalysisInfo getAnalysisInfoResult(){
        return analysisInfoResult;
    }
};