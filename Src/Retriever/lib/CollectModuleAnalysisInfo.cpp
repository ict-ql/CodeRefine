#include "CollectModuleAnalysisInfo.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Passes/PassBuilder.h"

#include "utils.h"

using namespace std;
using namespace llvm;

map<Instruction*, int> Inst2Idx;
string CollectModuleAnalysisInfoPass::getFuncIR(Function &F){
    
    string funcIRStr = "";
    raw_string_ostream OS(funcIRStr);
    F.print(OS);

    int idx = 0;
    for (Instruction &I : instructions(F)){
        Inst2Idx[&I] = idx;
        idx++;
    }
    // for (auto &BB: F){
    //     BB.printAsOperand(OS);
    //     for (auto &I: BB){
    //         I.print(OS);
    //     }
    // }
    // errs()<<funcIRStr<<"\n";
    // assert(0);
    return funcIRStr;
}

string CollectModuleAnalysisInfoPass::getBlockFrequencyInfo(BlockFrequencyInfo *BFI){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    BFI->print(OS);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getDemandedBitsInfo(DemandedBits *DB){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    DB->print(OS);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getDominatorTreeInfo(DominatorTree *DT){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    DT->print(OS);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getLoopAccessInfoManagerInfo(LoopAccessInfoManager *LAIs, LoopInfo *LI){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    for (Loop *TopLevelLoop : *LI){
        for (Loop *L : depth_first(TopLevelLoop)){
            (LAIs->getInfo(*L)).print(OS);
        }
    }
    return analysisResult;
}

static void printLoop(Loop *L, raw_string_ostream &OS, int Depth){
    OS.indent(Depth * 2);
    OS << "Loop S:"<< L->isLoopSimplifyForm();
    // OS << " hasInvariantOps:[";
    // for(auto BI = L->block_begin(), BE = L->block_end(); BI != BE; ++BI){
    //     for(auto II = (*BI)->begin(), IE = (*BI)->end(); II != IE; ++II){
    //         if(L->hasLoopInvariantOperands(&(*II))){
    //             (*II).printAsOperand(OS);
    //             OS<<",";
    //         }
    //     }
    // }
    // OS << "]";
    OS << " at depth " << L->getLoopDepth() << " containing: ";

    BasicBlock *H = L->getHeader();
    for (unsigned i = 0; i < L->getBlocks().size(); ++i) {
        BasicBlock *BB = L->getBlocks()[i];
        if (i)
            OS << ",";
        BB->printAsOperand(OS, false);

        if (BB == H)
            OS << "<header>";
        if (L->isLoopLatch(BB))
            OS << "<latch>";
        if (L->isLoopExiting(BB))
            OS << "<exiting>";
    }

    OS << "\n";
    for (auto I = L->begin(), E = L->end(); I != E; ++I)
        printLoop(*I, OS, Depth + 2);
}

string CollectModuleAnalysisInfoPass::getLoopInfo(LoopInfo *LI){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    // LI->print(OS);
    for (Loop *L: LI->getTopLevelLoops()){
        printLoop(L, OS, 0);
    }
    // errs()<<analysisResult<<"\n";
    // assert(0);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getMemorySSAInfo(MemorySSA *MSSA){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    MSSA->print(OS);
    return analysisResult;
}

/**
 * @brief part of result of ScalarEvolution->print
*/
string CollectModuleAnalysisInfoPass::getScalarEvolutionInfo(Function &F, ScalarEvolution *SE, LoopInfo *LI){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    
    OS << "Classifying expressions for: ";
    F.printAsOperand(OS, false);
    OS << "\n";

    for(Instruction &I: instructions(F)){ // only focus on loop, otherwise will need long time to analysis
        const Loop *L = LI->getLoopFor(I.getParent());
        if(L){
            if(SE->isSCEVable(I.getType()) && !isa<CmpInst>(I)){
                OS<<Inst2Idx[&I]<<" -> ";
                const SCEV *SV = SE->getSCEV(&I);
                SV->print(OS);
                OS<<"\n";
            }
        }
        // if(SE->isSCEVable(I.getType()) && !isa<CmpInst>(I)){
        //     OS<<Inst2Idx[&I]<<" -> ";
        //     const SCEV *SV = SE->getSCEV(&I);
        //     SV->print(OS);
        //     OS<<"\n";
        // }
    }

    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getCFGRelatedResultInfo(Function &F, BlockFrequencyInfo *BFI){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    if(F.isDeclaration())
        return analysisResult;
    for(BasicBlock &BB: F){
        // OS<<"%"<<BB.getName();
        BB.printAsOperand(OS);
        if(BB.isEntryBlock())
            OS<<" <isEntry> ";
        OS<<" BlockFreq"+to_string(BFI->getBlockFreq(&BB).getFrequency());
        OS<<":\n";
        OS<<"\tPredecessors:";
        for(auto itr = pred_begin(&BB), ie = pred_end(&BB); itr != ie; ++itr){
            BasicBlock *pred_BB = *itr;
            pred_BB->printAsOperand(OS); 
            // OS<<"%"<<pred_BB->getName();
            OS<<",";
        }
        OS<<"\n\tSuccessors:";
        for(auto itr = succ_begin(&BB), ie = succ_end(&BB); itr != ie; ++itr){
            BasicBlock *succ_BB = *itr;
            succ_BB->printAsOperand(OS); 
            // OS<<"%"<<succ_BB->getName();
            OS<<",";
        }
        OS<<"\n";
    }
    // OS<<"[";
    // for(BasicBlock &BB: F){
    //     OS<<"('";
    //     BB.printAsOperand(OS);
    //     OS<<"',";
    //     OS<<" {'Predecessors':[";
    //     for(auto itr = pred_begin(&BB), ie = pred_end(&BB); itr != ie; ++itr){
    //         BasicBlock *pred_BB = *itr;
    //         OS<<"'"; pred_BB->printAsOperand(OS); OS<<"',";
    //     }
    //     OS<<"], 'Successors':[";
    //     for(auto itr = succ_begin(&BB), ie = succ_end(&BB); itr != ie; ++itr){
    //         BasicBlock *succ_BB = *itr;
    //         OS<<"'"; succ_BB->printAsOperand(OS); OS<<"',";
    //     }
    //     OS<<"]";
    //     OS<<", 'Attribute':[";
    //     if(BB.isEntryBlock())
    //         OS<<"'isEntry',";
    //     // Loop *loop = LI->getLoopFor(&BB);
    //     // if(loop){
    //     //     OS<<"'<inloop>',";
    //     //     if(loop->getHeader() == &BB)
    //     //         OS<<"'<header>',";
    //     //     if(loop->isLoopExiting(&BB))
    //     //         OS<<"'<exiting>',";
    //     //     if(loop->isLoopLatch(&BB))
    //     //         OS<<"'<latch>',";
    //     // }
    //     OS<<"]";
    //     OS<<", 'BlockFreq':"+to_string(BFI->getBlockFreq(&BB).getFrequency());
    //     OS<<"}), ";
    // }
    // OS<<"]";
    // errs()<<analysisResult<<"\n";
    // assert(0);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getInstructionUserInfo(Instruction *I){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    for(auto U: I->users()){
        if (auto UI = dyn_cast<Instruction>(U)){
            OS<<"'"; UI->printAsOperand(OS); OS<<"', ";
        }
    }
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getInstructionOperandInfo(Instruction *I){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    for(auto itr = I->op_begin(), ie = I->op_end(); itr != ie; itr++){
        OS<<"'"; itr->get()->printAsOperand(OS); OS<<"', ";
    }
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getInstructionStr(Instruction *I){
    string str = "";
    raw_string_ostream OS(str);
    I->print(OS);
    return str;
}

string CollectModuleAnalysisInfoPass::getSCEVInfo(ScalarEvolution *SE, Instruction *I){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    auto scev = SE->getSCEV(I);
    switch(scev->getSCEVType()){
    case scUnknown:
    case scConstant:
        return ""; 
    default:
        scev->print(OS);
    }
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getDFRelatedResultInfo(Function &F,
                                                                MemorySSA *MSSA,
                                                                LoopAccessInfoManager *LAIs, 
                                                                LoopInfo *LI,
                                                                ScalarEvolution *SE){
    map<Instruction*, string> instr2DFResult;
    // init
    for(auto &BB: F){
        for(auto &I: BB){
            instr2DFResult[&I] = "{";
        }
    }

    // prepare each instruction's user and use info
    // for(auto &BB: F){
    //     for(auto &I: BB){
    //         string userInfo = getInstructionUserInfo(&I);
    //         string useInfo = getInstructionOperandInfo(&I);
    //         instr2DFResult[&I] += "'userInfo': ["+userInfo+"], 'useInfo': ["+useInfo+"], ";
    //     }
    // }

    // prepare dep (only add the info after dep?) info
    for (Loop *TopLevelLoop : *LI){
        for (Loop *L : depth_first(TopLevelLoop)){
            auto DepChecker = LAIs->getInfo(*L).getDepChecker();
            if(auto *Dependences = DepChecker.getDependences()){
                for(const auto &Dep: *Dependences){
                    Instruction *sourceI = Dep.getSource(LAIs->getInfo(*L));
                    Instruction *destI = Dep.getDestination(LAIs->getInfo(*L));
                    string dest = getInstructionStr(destI);
                    string depType = MemoryDepChecker::Dependence::DepName[Dep.Type];
                    if(instr2DFResult[sourceI].find("'Dep':") != string::npos){
                        instr2DFResult[sourceI].pop_back(); // delete ']'
                        instr2DFResult[sourceI] += "('"+dest+"', '<"+depType+">'), ]";
                    }
                    else
                        instr2DFResult[sourceI] += "'Dep': [('"+dest+"', '<"+depType+">'), ]";
                }
            }
        }
    }

    // prepare SE info: the info may has some constant that we cannot get it from FunctionIR, so we delete it.
    // for(auto &BB: F){
    //     for(auto &I: BB){
    //         string scevStr = getSCEVInfo(SE, &I);
    //         if(scevStr != "")
    //             instr2DFResult[&I] += "'SCEV': '"+scevStr+"'";
    //     }
    // }

    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    string MSSA_str = getMemorySSAInfo(MSSA);
    // print all
    OS<<"{'MSSA':'"<<MSSA_str<<"', 'InstrInfo':[";
    for(auto &BB: F){ // print in IR order
        for(auto &I: BB){
            // if(instr2DFResult[&I] != "{"){
            OS<<"('"; I.print(OS); OS<<"', "<<instr2DFResult[&I]<<"}), ";
            // }
        }
    }
    // for(auto &KV: instr2DFResult){
    //     Instruction* I = KV.first;
    //     if(KV.second != "{"){
    //         OS<<"('"; I->print(OS); OS<<"', "<<KV.second<<"}), ";
    //     }
    // }
    OS<<"]}";
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getAliasAnalysisResultInfo(Function &F,
                                                                    AAResults *AA){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    BatchAAResults BatchAA(*AA);
    AliasSetTracker Tracker(BatchAA);
    OS << "Alias sets for function '" << F.getName() << "':\n";
    for (Instruction &I : instructions(F))
        Tracker.add(&I);
    Tracker.print(OS);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getInstructionAttrResultInfo(Function &F){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    for (Instruction &I : instructions(F)){
        OS<<Inst2Idx[&I]<<":";
        if(CallBase *CB = dyn_cast<CallBase>(&I)){
            if(CB->doesNotAccessMemory())
                OS<<"1";
            else
                OS<<"0";
            if(CB->onlyReadsMemory())
                OS<<"1";
            else
                OS<<"0";
        }else{
            OS<<"//";
        }

        if(GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(&I)){
            if(GEPI->hasAllConstantIndices())
                OS<<"1";
            else
                OS<<"0";
            if(GEPI->isInBounds())
                OS<<"1";
            else
                OS<<"0";
        }
        else{
            OS<<"//";
        }

        if(I.getType()->isPointerTy()){
            Value *tmpV = getUnderlyingObject(&I);
            tmpV->printAsOperand(OS);
        }
        else{
            OS<<"/";
        }
        
        if(isa<OverflowingBinaryOperator>(I)){
            if(I.hasNoSignedWrap())
                OS<<"1";
            else
                OS<<"0";
            if(I.hasNoUnsignedWrap())
                OS<<"1";
            else
                OS<<"0";
        }
        else{
            OS<<"//";
        }
        if(I.isAtomic())
            OS<<"1";
        else
            OS<<"0";
        if(I.isCommutative())
            OS<<"1";
        else
            OS<<"0";
        if(I.isLifetimeStartOrEnd())
            OS<<"1";
        else
            OS<<"0";
        if(I.mayHaveSideEffects())
            OS<<"1";
        else
            OS<<"0";
        if(I.mayReadFromMemory())
            OS<<"1";
        else
            OS<<"0";
        if(I.mayThrow())
            OS<<"1";
        else
            OS<<"0";
        if(I.mayWriteToMemory())
            OS<<"1";
        else
            OS<<"0";
        if(isGuaranteedToTransferExecutionToSuccessor(&I))
            OS<<"1";
        else
            OS<<"0";
        //@TODO isGuaranteedToTransferExecutionToSuccessor

        OS<<to_string(I.getNumUses())<<"\n";
    }
    // OS<<"[";
    // for (Instruction &I : instructions(F)){
    //     OS<<"('"<<Inst2Idx[&I]<<"',";
    //     OS<<" {'Attr':[";
    //     if(CallBase *CB = dyn_cast<CallBase>(&I)){
    //         if(CB->doesNotAccessMemory())
    //             OS<<"1,";
    //         else
    //             OS<<"0,";
    //         if(CB->onlyReadsMemory())
    //             OS<<"1,";
    //         else
    //             OS<<"0,";
    //     }else{
    //         OS<<"-1,-1,";
    //     }

    //     if(GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(&I)){
    //         if(GEPI->hasAllConstantIndices())
    //             OS<<"1,";
    //         else
    //             OS<<"0,";
    //         if(GEPI->isInBounds())
    //             OS<<"1,";
    //         else
    //             OS<<"0,";
    //     }
    //     else{
    //         OS<<"-1,-1,";
    //     }
    //     OS<<"'";
    //     if(I.getType()->isPointerTy()){
    //         Value *tmpV = getUnderlyingObject(&I);
    //         tmpV->printAsOperand(OS);
    //     }
    //     else{
    //         OS<<"NULL";
    //     }
    //     OS<<"',";
    //     if(isa<OverflowingBinaryOperator>(I)){
    //         if(I.hasNoSignedWrap())
    //             OS<<"1,";
    //         else
    //             OS<<"0,";
    //         if(I.hasNoUnsignedWrap())
    //             OS<<"1,";
    //         else
    //             OS<<"0,";
    //     }
    //     else{
    //         OS<<"-1,-1,";
    //     }
    //     if(I.isAtomic())
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     if(I.isCommutative())
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     if(I.isLifetimeStartOrEnd())
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     if(I.mayHaveSideEffects())
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     if(I.mayReadFromMemory())
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     if(I.mayThrow())
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     if(I.mayWriteToMemory())
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     if(isGuaranteedToTransferExecutionToSuccessor(&I))
    //         OS<<"1,";
    //     else
    //         OS<<"0,";
    //     //@TODO isGuaranteedToTransferExecutionToSuccessor

    //     OS<<to_string(I.getNumUses())<<",";
    //     OS<<"]";
    //     OS<<"}), ";
    // }
    // OS<<"]";
    // errs()<<analysisResult<<"\n";
    // assert(0);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getGlobalVarInfo(Function &F){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);

    set<const GlobalValue*> Globals;
    set<const StructType*> STs;
    for (const BasicBlock &BB : F){
        for (const Instruction &I : BB){
            if(const StructType* ST = dyn_cast<StructType>(I.getType()))
                STs.insert(ST);
            if(const GetElementPtrInst* GEPI = dyn_cast<GetElementPtrInst>(&I)){
                if(const StructType* ST = dyn_cast<StructType>(GEPI->getSourceElementType()))
                    STs.insert(ST);
                if(const StructType* ST = dyn_cast<StructType>(GEPI->getResultElementType()))
                    STs.insert(ST);
            }
            for (const Value *Op : I.operands()){
                if (const GlobalValue* G = dyn_cast<GlobalValue>(Op))
                    Globals.insert(G);
                if(const StructType* ST = dyn_cast<StructType>(Op->getType()))
                    STs.insert(ST);
            }
        }
    }
    for (auto T : STs) {
        T->print(OS);
        OS << '\n';
    }

    for(auto GV: Globals){
        if(GV->isDeclaration())
            OS<<"1";
        else
            OS<<"0";
        if(GV->isInterposable())
            OS<<"1";
        else
            OS<<"0";
        OS<<" ";
        GV->print(OS);
        OS<<"\n";
    }

    // errs()<<analysisResult;
    // assert(0);
    return analysisResult;
}

string CollectModuleAnalysisInfoPass::getDataLayoutResultInfo(Function &F){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    OS<<F.getParent()->getDataLayoutStr();
    return analysisResult;
}

/**
 * @brief get the declare funcs that the input function depends.
*/
string CollectModuleAnalysisInfoPass::getFunctionDeclareInfo(Function &F){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);

    set<const Function*> declareFuncs;
    for (const BasicBlock &BB : F){
        for (const Instruction &I : BB){
            if(const CallBase *CB = dyn_cast<CallBase>(&I)){
                auto *cf = CB->getCalledFunction();
                if(cf && cf->isDeclaration())
                    declareFuncs.insert(cf);
            }
        }
    }

    for(auto df: declareFuncs){
        df->print(OS);
    }
    // errs()<<analysisResult<<"\n";
    // assert(0);
    return analysisResult;
}

PreservedAnalyses CollectModuleAnalysisInfoPass::run(Function &F, FunctionAnalysisManager &AM) {
    if(F.isDeclaration())
        return PreservedAnalyses::all();
    auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
    auto *DB = &AM.getResult<DemandedBitsAnalysis>(F);
    auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
    auto *LAIs = &AM.getResult<LoopAccessAnalysis>(F);
    auto *LI = &AM.getResult<LoopAnalysis>(F);
    auto *MSSA = &AM.getResult<MemorySSAAnalysis>(F);
    auto *SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
    auto *AA = &AM.getResult<AAManager>(F);
    // errs()<<"1111\n";
    analysisInfoResult.FuncIR = getFuncIR(F);
    if(analysisInfoResult.FuncIR.size() >= 20000) // we only focus on the func that can be put in llama ctx
        return PreservedAnalyses::all();
    // errs()<<"2222\n";
    analysisInfoResult.CFGRelatedResult = getCFGRelatedResultInfo(F, BFI);
    // errs()<<"3333\n";
    // analysisInfoResult.DFRelatedResult = getDFRelatedResultInfo(F, MSSA, LAIs, LI, SE);
    analysisInfoResult.LoopInfoResult = getLoopInfo(LI);
    // errs()<<"4444\n";
    analysisInfoResult.AAResult = getAliasAnalysisResultInfo(F, AA);
    // errs()<<"5555\n";
    analysisInfoResult.InstResult = getInstructionAttrResultInfo(F);
    // errs()<<"6666\n";
    analysisInfoResult.DominatorTreeAnalysisResult = getDominatorTreeInfo(DT);
    // errs()<<"7777\n";
    analysisInfoResult.ScalarEvolutionAnalysisResult = getScalarEvolutionInfo(F, SE, LI);
    // errs()<<"8888\n";
    // in Train Data Mod, we donnot need datalayout and GlobalVarResult
    // analysisInfoResult.DataLayoutResult = getDataLayoutResultInfo(F);
    // errs()<<"9999\n";
    // analysisInfoResult.GlobalVarResult = getGlobalVarInfo(F);
    // errs()<<"0000\n";

    // // analysisInfoResult.DeclareFuncResult = getFunctionDeclareInfo(F);

    // errs()<<"analysisInfoResult.DominatorTreeAnalysisResult:\n"<<analysisInfoResult.DominatorTreeAnalysisResult<<"\n";
    // errs()<<"analysisInfoResult.ScalarEvolutionAnalysisResult:\n"<<analysisInfoResult.ScalarEvolutionAnalysisResult<<"\n";

    // CollectModuleAnalysisInfoLegacy::Func2AnalysisInfo[&F] = {{"func_ir", analysisInfoResult.FuncIR}, {"cfg_info", analysisInfoResult.CFGRelatedResult}, {"loop_info", analysisInfoResult.LoopInfoResult}, {"alias_info", analysisInfoResult.AAResult}, {"inst_info", analysisInfoResult.InstResult}, {"se_info", analysisInfoResult.ScalarEvolutionAnalysisResult}, {"dt_info", analysisInfoResult.DominatorTreeAnalysisResult}, {"data_layout_info", analysisInfoResult.DataLayoutResult}, {"global_var_info", analysisInfoResult.GlobalVarResult}, {"declare_func_info", analysisInfoResult.DeclareFuncResult}};
    CollectModuleAnalysisInfoPass::Func2AnalysisInfo[&F] = analysisInfoResult;

    return PreservedAnalyses::all();
}

void runCollectModuleAnalysisInfoTest(string InputFilePath){
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

    std::unique_ptr<Module> M = getLLVMIRFromFile(InputFilePath);
    MPM.run(*M, MAM);
}

map<Function*, AnalysisInfo> runToCollectModuleAnalysisInfo(std::unique_ptr<Module> &M){
    CollectModuleAnalysisInfoPass::Func2AnalysisInfo.clear();

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
    MPM.run(*M, MAM);
    return CollectModuleAnalysisInfoPass::Func2AnalysisInfo;
}

map<Function*, AnalysisInfo> CollectModuleAnalysisInfoPass::Func2AnalysisInfo;