#include "splitLongBBForPrompt.h"
#include "Scope.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <set>
#include <list>
#include <algorithm>
#include <string>
#include <json/json.h>
#include<regex>

#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"

using namespace std;
using namespace llvm;

bool forVec = false;

class PromptWithSmallerBBInfo{
private:
    Function *beforeF;
    Function *afterF;

    Scope *beforeRootScope;
    Scope *originAfterRootScope;

    string originAfterFStr;
    string originAfterFCFG;

    map<BasicBlock*, string> AfterBB2LeastBeforeBBs;
    map<BasicBlock*, string> AfterBB2MostBeforeBBs;
    map<BasicBlock*, BasicBlock*> BB2OriginBB;

    map<pair<unsigned, unsigned>, vector<Instruction*>> loc2OriginInst;

    // the following function can be moved to utils.cpp
    string getFunctionIR(Function *F);

    string getCFGOfFunc(Function *F);
    vector<Value*> getBBIn(BasicBlock *BB);
    vector<Value*> getBBOut(BasicBlock *BB);
    map<string, string> getBBName2BBIR(Function *F);
    map<string, string> getBBName2BBIn(Function *F);
    map<string, string> getBBName2BBOut(Function *F);

    vector<Scope*> getAllScopeByChildToFather(Scope *root);
    BasicBlock* getWhereBBFrom(BasicBlock *BB){
        if(BB2OriginBB.find(BB) != BB2OriginBB.end()){
            return BB2OriginBB[BB];
        }
        return NULL;
    }
    BasicBlock* getWhereBBFrom(string targetBBName){
        for(auto &KV: BB2OriginBB){
            BasicBlock *BB =  KV.first;
            string BBName = "";
            raw_string_ostream NameOS(BBName);
            BB->printAsOperand(NameOS);
            if(BBName == targetBBName){
                return KV.second;
            }
        }
        return NULL;
    }

    void printLoop(Loop *L, raw_string_ostream &OS, int Depth){
        OS.indent(Depth * 2);
        OS << "Loop S:"<< L->isLoopSimplifyForm();
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

    string getLoopInfo(Function *F){
        DominatorTree DT(*F);
        LoopInfo LI(DT);

        string analysisResult = "";
        raw_string_ostream OS(analysisResult);
        for (Loop *L: LI.getTopLevelLoops()){
            printLoop(L, OS, 0);
        }
        return analysisResult;
    }
public:
    PromptWithSmallerBBInfo(Function *beforeF, Function *afterF, int InstructionThreshold):
        beforeF(beforeF), afterF(afterF) {
        // rename the BBname
        for(BasicBlock &BB: *afterF){
            BB.setName("B");
        }

        originAfterRootScope = constructScope(afterF);

        // delete dbg Instruction
        for(auto &BB: *afterF){
            for (auto it = BB.begin(), end = BB.end(); it != end; ){
                if((*it).isDebugOrPseudoInst()){
                    it = (*it).eraseFromParent();
                }else{
                    ++it;
                }
            }
        }
        originAfterFStr = getFunctionIR(afterF);
        originAfterFCFG = getCFGOfFunc(afterF);

        // to prevent the opt BB IR too long, split it
        for(BasicBlock &BB: *afterF){
            BasicBlock *currentBB = &BB;
            while(currentBB->size() > InstructionThreshold){
                auto Inst = currentBB->begin();
                std::advance(Inst, InstructionThreshold);
                // errs()<<"before:\n";
                // currentBB->print(errs());

                currentBB = currentBB->splitBasicBlockBefore(Inst, ".split")->getSingleSuccessor();
                assert(currentBB && "ERROR: after splitting, the origin BB must has only one successor!\n");
                BB2OriginBB[currentBB->getSinglePredecessor()] = getWhereBBFrom(currentBB)?
                                         getWhereBBFrom(currentBB):
                                         currentBB;

                // errs()<<"after:\n";
                // currentBB->print(errs());
            }
        }
        // // rename the BBname
        // for(BasicBlock &BB: *afterF){
        //     BB.setName("B");
        // }

        beforeRootScope = constructScope(beforeF);
        // delete dbg Instruction
        for(auto &BB: *beforeF){
            for (auto it = BB.begin(), end = BB.end(); it != end; ){
                if((*it).isDebugOrPseudoInst()){
                    it = (*it).eraseFromParent();
                }else{
                    ++it;
                }
            }
        }
        if(!beforeRootScope || !originAfterRootScope){
            errs()<<"WARNING: COMPILE "<<beforeF->getName().str()<<" WITH -g!\n";
            beforeRootScope = NULL;
            originAfterRootScope = NULL;
        }
        // assert(beforeRootScope && originAfterRootScope && "ERROR: all ir must be compile by `-g`!\n");
        // errs()<<"BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!\n"<<beforeRootScope->toString()<<"\n";
        // errs()<<"AFTER!!!!!!!!!!!!!!!!!!!!!!!!!\n"<<afterRootScope->toString()<<"\n";

        for (auto &I : instructions(beforeF)) {
            if(I.getDebugLoc()){
                unsigned line = I.getDebugLoc().getLine();
                unsigned col = I.getDebugLoc().getCol();
                pair<unsigned, unsigned> p = make_pair(line, col);
                if(I.getNumUses() == 0){
                    continue;
                }
                if(loc2OriginInst.find(p) != loc2OriginInst.end()){
                    loc2OriginInst[p].push_back(&I);
                }else{
                    loc2OriginInst[p] = {&I};
                }
            }
        }
    }

    string getUnoptScope2OptScope();

    string getBeforeFuncIR(){
        return getFunctionIR(beforeF);
    }

    string getOriginAfterFuncIR(){
        return originAfterFStr;
    }

    string getAfterFuncIRWithSmallerBB(){
        return getFunctionIR(afterF);
    }

    string getBeforeCFG(){
        return getCFGOfFunc(beforeF);
    }

    string getOriginAfterFuncCFG(){
        return originAfterFCFG;
    }

    string getAfterCFGWithSmallerBB(){
        return getCFGOfFunc(afterF);
    }

    map<string, string> getAfterBBName2BB(){
        return getBBName2BBIR(afterF);
    }

    map<string, string> getBeforeBBName2BB(){
        return getBBName2BBIR(beforeF);
    }

    map<string, string> getAfterBBName2BBIn(){
        return getBBName2BBIn(afterF);
    }

    map<string, string> getAfterBBName2BBOut(){
        return getBBName2BBOut(afterF);
    }

    string getBeforeFuncLoopInfo(){
        return getLoopInfo(beforeF);
    }

    string getLeastBeforeBBsofTargetBB(string afterBBName){
        BasicBlock *pBB = getWhereBBFrom(afterBBName);
        if(pBB){
            for(auto &KV: AfterBB2LeastBeforeBBs){
                BasicBlock *BB =  KV.first;
                if(BB == pBB){
                    return KV.second;
                }
            }
        }else{
            for(auto &KV: AfterBB2LeastBeforeBBs){
                BasicBlock *BB =  KV.first;
                string BBName = "";
                raw_string_ostream NameOS(BBName);
                BB->printAsOperand(NameOS);
                if(BBName == afterBBName){
                    return KV.second;
                }
            }
        }

        return beforeRootScope->toString();
        
        // errs()<<afterBBName<<"\n";
        // errs()<<originAfterRootScope->toString()<<"\n";
        // for(auto &KV: BB2OriginBB){
        //     KV.first->printAsOperand(errs());
        //     errs()<<"\tp:";
        //     KV.second->printAsOperand(errs());
        //     errs()<<"\n";
        // }
        // afterF->print(errs());
        // assert(0 && "ERROR: there must be a scope corresponding to the after BB!\n");
        // return ""; 
    }

    
    string getMostBeforeBBsofTargetBB(string afterBBName){
        BasicBlock *pBB = getWhereBBFrom(afterBBName);
        if(pBB){
            for(auto &KV: AfterBB2MostBeforeBBs){
                BasicBlock *BB =  KV.first;
                if(BB == pBB){
                    return KV.second;
                }
            }
        }else{
            for(auto &KV: AfterBB2MostBeforeBBs){
                BasicBlock *BB =  KV.first;
                string BBName = "";
                raw_string_ostream NameOS(BBName);
                BB->printAsOperand(NameOS);
                // errs()<<BBName<<" t:"<<afterBBName<<"\n";
                if(BBName == afterBBName){
                    return KV.second;
                }
            }
        }

        // errs()<<afterBBName<<"\n";
        // errs()<<originAfterRootScope->toString()<<"\n";
        // for(auto &KV: BB2OriginBB){
        //     KV.first->printAsOperand(errs());
        //     errs()<<"\tp:";
        //     KV.second->printAsOperand(errs());
        //     errs()<<"\n";
        // }

        // assert(0 && "ERROR: there must be a scope corresponding to the after BB!\n");
        // return ""; 
        return beforeRootScope->toString();
    }

    string getTheLastInstructionName(string targetBBName, Function *F){
        string lastInstr = "";

        for(BasicBlock &BB: *F){
            string BBName = "";
            raw_string_ostream NameOS(BBName);
            BB.printAsOperand(NameOS);
            if(BBName == targetBBName){
                if(lastInstr == ""){
                    for(auto &arg: F->args()){
                        string argName = "";
                        raw_string_ostream argNameOS(argName);
                        arg.printAsOperand(argNameOS);
                        lastInstr = argName;
                    }
                }
                return lastInstr;
            }
            for(Instruction &I: BB){
                string tmpInst = "";
                raw_string_ostream InstNameOS(tmpInst);
                I.printAsOperand(InstNameOS);
                if(tmpInst != "void <badref>"){
                    // tmpInst = "";
                    // I.print(InstNameOS);
                    lastInstr = tmpInst;
                }
            }
        }

        assert(0 && "ERROR: the BB cannot be found in F!\n");
        return lastInstr;
    }

    map<string, string> getAfterBB2OriginInst(){
        vector<Scope*> beforeScopes = getAllScopeByChildToFather(beforeRootScope);
        map<pair<unsigned, unsigned>, Scope*> loc2Scope;
        for(Scope* s: beforeScopes){
            unsigned line = s->getLine();
            unsigned col = s->getColumn();
            if(line == 0 && col == 0)
                continue;
            auto p = make_pair(line, col);
            if(loc2Scope.find(p) == loc2Scope.end())
                loc2Scope[p] = s;
        }

        map<string, string> res;
        for(BasicBlock &BB: *afterF){
            string BBName = "";
            raw_string_ostream NameOS(BBName);
            BB.printAsOperand(NameOS);

            vector<pair<unsigned, unsigned>> locs;
            map<pair<unsigned, unsigned>, vector<Instruction*>> loc2AfterIs;
            vector<SelectInst*> selectIs;
            for(Instruction &I: BB){
                if(I.getDebugLoc()){
                    if(SelectInst* SI = dyn_cast<SelectInst>(&I)){
                        selectIs.push_back(SI);
                        continue;
                    }
                    auto p = make_pair(I.getDebugLoc().getLine(),I.getDebugLoc().getCol());
                    if(find(locs.begin(), locs.end(), p) == locs.end()){
                        locs.push_back(p);
                        loc2AfterIs[p] = {&I};
                    }else{
                        loc2AfterIs[p].push_back(&I);
                    }
                    
                }
            }

            string ans = "";
            raw_string_ostream ansOS(ans);
            for(auto p: locs){
                // errs()<<p.first<<" "<<p.second<<"\n";
                if(loc2OriginInst.find(p) == loc2OriginInst.end())
                    continue;
                bool afterIContainsSSAName = false;
                for(Instruction *tmpI: loc2AfterIs[p]){
                    if(tmpI->getNumUses() != 0){
                        tmpI->printAsOperand(ansOS);
                        ansOS<<",";
                        afterIContainsSSAName = true;
                    }
                }
                if(!afterIContainsSSAName)
                    continue;
                
                ansOS<<" Corresponding to instr above:{\n";
                for(Instruction *tmpI: loc2OriginInst[p]){
                    // tmpI->print(errs());
                    // errs()<<"\n";
                    if(tmpI->getNumUses() != 0){
                        tmpI->print(ansOS);
                        ansOS<<"\n";
                    }
                }
                ansOS<<"}\n";
            }

            map<Scope*, vector<SelectInst*>> scope2SIs;
            for(auto SI: selectIs){
                if(DILexicalBlock *scope = dyn_cast<DILexicalBlock>(SI->getDebugLoc().getScope())){
                    unsigned line = scope->getLine();
                    unsigned col = scope->getColumn();
                    auto p = make_pair(line, col);
                    if(loc2Scope.find(p) != loc2Scope.end()){
                        Scope *s = loc2Scope[p];
                        if(scope2SIs.find(s) != scope2SIs.end()){
                            scope2SIs[s].push_back(SI);
                        }else{
                            scope2SIs[s] = {SI};
                        }
                    }
                }
            }

            for(auto &KV: scope2SIs){
                for(auto SI: KV.second){
                    SI->printAsOperand(ansOS);
                    ansOS<<",";
                }
                ansOS<<" Corresponding to BB above:{";
                for(auto BB: KV.first->getBBs()){
                    BB->printAsOperand(ansOS);
                    ansOS<<",";
                }
                ansOS<<"}\n";
            }
            res[BBName] = ans;
        }
        return res;
    }

    Scope* getBeforeRootScope(){
        return beforeRootScope;
    }

    Scope* getAfterRootScope(){
        return originAfterRootScope;
    }

    Function* getAfterFunction(){
        return afterF;
    }

    Function* getBeforeFunction(){
        return beforeF;
    }
    
    string getLocOfBB(string BBName){
        string ans = "";
        BasicBlock *BB = NULL;
        for(auto &tmpB: *afterF){
            string tName = "";
            raw_string_ostream NameOS(tName);
            tmpB.printAsOperand(NameOS);
            if(tName == BBName){
                BB = &tmpB;
                break;
            }
        }
        if(!BB)
            return ans;
        for(auto &I: *BB){
            if(I.getDebugLoc()){
                unsigned line = I.getDebugLoc().getLine();
                unsigned col = I.getDebugLoc().getCol();
                ans += "<"+to_string(line)+","+to_string(col)+">,";
            }
        }
        return ans;
    }

    string getLocOfInstr(Instruction *I){
        string ans = "";
        if(I->getDebugLoc()){
            unsigned line = I->getDebugLoc().getLine();
            unsigned col = I->getDebugLoc().getCol();
            ans += "<"+to_string(line)+","+to_string(col)+">";
        }
        return ans;
    }
};

vector<Scope*> PromptWithSmallerBBInfo::getAllScopeByChildToFather(Scope *root){
    set<Scope*> innermostScopes = root->getInnermostScopes();

    vector<Scope*> ans(innermostScopes.begin(), innermostScopes.end());
    for(vector<Scope*>::size_type i=0; i<ans.size(); ++i){
        if(ans[i]->getParent()){
            if(find(ans.begin(), ans.end(), ans[i]->getParent()) == ans.end()) // has not visited before
                ans.push_back(ans[i]->getParent());
        }
    }
    // if(ans.back() != root){
    //     beforeF->print(errs());
    // }
    // assert(ans.back() == root && "ERROR: the last element must be the `root`!\n");
    return ans;
}

string PromptWithSmallerBBInfo::getUnoptScope2OptScope(){
    map<Scope*, Scope*> after2Before;
    vector<Scope*> beforeScopes = getAllScopeByChildToFather(beforeRootScope);
    vector<Scope*> afterScopes = getAllScopeByChildToFather(originAfterRootScope);

    for(Scope* as: afterScopes){
        unsigned aline = as->getLine();
        unsigned acol = as->getColumn();
        string aFileName = as->getFileName();
        for(Scope* bs: beforeScopes){
            unsigned bline = bs->getLine();
            unsigned bcol = bs->getColumn();
            string bFileName = bs->getFileName();
            if(aline == bline && acol == bcol && aFileName == bFileName){
                after2Before[as] = bs;
                break;
            }
        }
        
    }

    for(auto &KV: after2Before){
        string beforeStr = KV.second->toString();
        Scope* afterS = KV.first;
        for(auto BB: afterS->getBBs()){
            if(AfterBB2LeastBeforeBBs.find(BB) != AfterBB2LeastBeforeBBs.end()){
                if(AfterBB2LeastBeforeBBs[BB].size() > beforeStr.size())
                    AfterBB2LeastBeforeBBs[BB] = beforeStr;
                if(AfterBB2MostBeforeBBs[BB].size() < beforeStr.size())
                    AfterBB2MostBeforeBBs[BB] = beforeStr;
            }else{
                AfterBB2LeastBeforeBBs[BB] = beforeStr;
                AfterBB2MostBeforeBBs[BB] = beforeStr;
            }
        }
    }

    // for(auto &KV: after2Before){
    //     errs()<<"Before: "<<KV.first->toString()<<"\n";
    //     errs()<<"After: "<<KV.second->toString()<<"\n";
    // }
    map<string, string> before2After;
    before2After.clear();
    for(auto &KV: after2Before){
        string beforeStr = KV.second->toString();
        string afterStr = KV.first->toString();
        if(before2After.find(beforeStr) != before2After.end()){
            if(before2After[beforeStr].size() < afterStr.size())
                before2After[beforeStr] = afterStr;
        }
        else
            before2After[beforeStr] = afterStr;
    }

    vector<string> b2a;
    b2a.clear();
    for(auto &KV: before2After){
        b2a.push_back(KV.first + "->" + KV.second);
    }
    std::sort(b2a.begin(), b2a.end());

    string ans = "";
    for(string str: b2a){
        ans += str + "\n";
    }
    return ans;
}

map<string, string> PromptWithSmallerBBInfo::getBBName2BBIR(Function *F){
    map<string, string> res;
    for(BasicBlock &BB: *F){
        string BBName = "";
        raw_string_ostream NameOS(BBName);
        BB.printAsOperand(NameOS);

        string BBIR = "";
        raw_string_ostream IROS(BBIR);
        BB.print(IROS);
        res[BBName] = BBIR;
    }
    return res;
}

vector<Value*> PromptWithSmallerBBInfo::getBBOut(BasicBlock *BB){
    vector<Value*> ans;
    for(Instruction &I: *BB){
        for(auto v: I.users()){
            if(Instruction *tmpI = dyn_cast<Instruction>(v)){
                if(tmpI->getParent() != BB){
                    ans.push_back(&I);
                    break;
                }
                // if the instruction is used in phi, although use by the bb itself
                if(isa<PHINode>(v)){
                    ans.push_back(&I);
                    break;
                }
            }
        }
    }
    return ans;
}

vector<Value*> PromptWithSmallerBBInfo::getBBIn(BasicBlock *BB){
    set<Value*> defs;
    set<Value*> uses;
    vector<Value*> ins;
    for(Instruction &I: *BB){
        defs.insert(&I);
        for(Value *V: I.operands()){
            if(isa<Instruction>(V))
                uses.insert(V);
        }
    }

    for(Instruction &I: *BB){
        // add called function
        if(CallBase  *CB = dyn_cast<CallBase>(&I)){
            if(CB->getCalledFunction())
                uses.insert(CB->getCalledFunction());
        }
        
        for(Value *V: I.operands()){
            // add function argument
            if(isa<Argument>(V))
                uses.insert(V);
            // add global variable
            if(isa<GlobalValue>(V))
                uses.insert(V);
        }
    }

    set_difference(uses.begin(), uses.end(),
                    defs.begin(), defs.end(),
                    back_inserter(ins));

    return ins;
}

map<string, string> PromptWithSmallerBBInfo::getBBName2BBOut(Function *F){
    map<string, string> res;
    for(BasicBlock &BB: *F){
        vector<Value*> Ins = getBBOut(&BB);
        string BBName = "";
        raw_string_ostream NameOS(BBName);
        BB.printAsOperand(NameOS);

        string out_str = "";
        raw_string_ostream OS(out_str);
        for(Value *out: Ins){
            assert(isa<Instruction>(out) && "The out value must in `Instruction` class!\n");
            out->printAsOperand(OS);
            OS<<";";
        }
        res[BBName] = out_str;
    }
    return res;
}

map<string, string> PromptWithSmallerBBInfo::getBBName2BBIn(Function *F){
    map<string, string> res;
    for(BasicBlock &BB: *F){
        vector<Value*> Ins = getBBIn(&BB);
        string BBName = "";
        raw_string_ostream NameOS(BBName);
        BB.printAsOperand(NameOS);

        string in_str = "";
        raw_string_ostream OS(in_str);
        for(Value *in: Ins){
            if(isa<Function>(in)){
                OS<<"callee: ";
                in->printAsOperand(OS);
            }
            else if(isa<Argument>(in)){
                OS<<"funcArg: ";
                in->printAsOperand(OS);
            }
            else if(isa<GlobalValue>(in)){
                OS<<"globalVar: ";
                in->printAsOperand(OS);
            }
            else{
                Instruction *I = dyn_cast<Instruction>(in);
                assert(I && "ERROR: the BB in must be in type `callee`/`funcArg`/`globalVar`!\n");
                OS<<"inst in ";
                assert(I->getParent() && "ERROR: the Instruction from BB in must have parent!\n");
                I->getParent()->printAsOperand(OS);
                OS<<": ";
                in->print(OS);
                if(I->getDebugLoc()){
                    unsigned line = I->getDebugLoc().getLine();
                    unsigned col = I->getDebugLoc().getCol();
                    if(forVec){
                        OS<<"<"<<to_string(line)<<","<<to_string(col)<<">";
                        // errs()<<"<"<<to_string(line)<<","<<to_string(col)<<">";
                    }
                    auto p = make_pair(line, col);
                    if(loc2OriginInst.find(p) != loc2OriginInst.end()){
                        OS<<"\n";
                        I->printAsOperand(OS);
                        OS<<" Corresponding to instr above:[";
                        for(Instruction* tmpI: loc2OriginInst[p]){
                            if(tmpI->getNumUses() != 0){
                                tmpI->printAsOperand(OS);
                                OS<<",";
                            }
                        }
                        OS<<"]";
                    }
                }
            }
            OS<<"\n";
        }
        res[BBName] = in_str;
    }
    return res;
}

string PromptWithSmallerBBInfo::getFunctionIR(Function *F){
    assert(F && "`F` must not be NULL!\n");

    string ans = "";
    raw_string_ostream OS(ans);
    F->print(OS);
    return ans;
}

string PromptWithSmallerBBInfo::getCFGOfFunc(Function *F){
    string analysisResult = "";
    raw_string_ostream OS(analysisResult);
    if(F->isDeclaration())
        return analysisResult;
    for(BasicBlock &BB: *F){
        // OS<<"%"<<BB.getName();
        BB.printAsOperand(OS);
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
    return analysisResult;
}

map<string, Function*> getFuncName2FuncPtr(unique_ptr<Module> &ModuleIR){
    map<string, Function*> res;
    for(Function &F: *ModuleIR){
        if(F.isDeclaration())
            continue;
        res[F.getName().str()] = &F;
    }
    return res;
}

vector<PromptWithSmallerBBInfo> getPrompts(map<string, Function*> &beforeFuncName2FuncPtr,
                                            map<string, Function*> &afterFuncName2FuncPtr,
                                            int InstructionThreshold){
    vector<PromptWithSmallerBBInfo> records;
    for(auto &KV: afterFuncName2FuncPtr){
        string funcName = KV.first;
        if(beforeFuncName2FuncPtr.find(funcName) == beforeFuncName2FuncPtr.end()){
            errs()<<funcName<<"\n";
            records.clear(); // there is a module pass that we cannot deal with
            break;
        }
        records.push_back(PromptWithSmallerBBInfo(beforeFuncName2FuncPtr[funcName], KV.second, InstructionThreshold));
    }
    return records;
}

bool startsWith(const std::string& fullString, const std::string& startingSubstring) {
    if (startingSubstring.size() > fullString.size()) return false;
    return fullString.compare(0, startingSubstring.size(), startingSubstring) == 0;
}

bool isFileExists(string name) {
    ifstream f(name.c_str());
    return f.good();
}

void writeIntoJson(vector<PromptWithSmallerBBInfo> &prompts, ofstream &outfile){
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    Json::StreamWriter* writer = builder.newStreamWriter();
    for(auto P: prompts){
        if(!P.getBeforeRootScope() || !P.getAfterRootScope())
            continue;
        Json::Value root;
        
        DIFile &difile = *(P.getBeforeFunction()->getParent()->debug_compile_units_begin())->getFile();
        // errs()<<difile.getDirectory().str()<<"/"
        //         <<difile.getFilename().str()<<"\n";
        if(isFileExists(difile.getFilename().str()))
            root["sourceFile"] = difile.getFilename().str();
        else if(isFileExists(difile.getDirectory().str()+"/"+difile.getFilename().str()))
            root["sourceFile"] = difile.getDirectory().str()+"/"+difile.getFilename().str();
        else
            errs()<<"WARNING!!!!!!!\n";
        root["beforeFuncIR"] = P.getBeforeFuncIR();
        root["originAfterFuncIR"] = P.getOriginAfterFuncIR();
        root["originAfterFuncCFG"] = P.getOriginAfterFuncCFG();
        root["afterFuncIRWithSmallerBB"] = P.getAfterFuncIRWithSmallerBB();
        root["beforeCFG"] = P.getBeforeCFG();
        root["afterCFGWithSmallerBB"] = P.getAfterCFGWithSmallerBB();
        root["beforeBB2afterBB"] = P.getUnoptScope2OptScope();
        root["beforeFuncLoopInfo"] = P.getBeforeFuncLoopInfo();
        map<string, string> beforeBBName2BB = P.getBeforeBBName2BB();
        for(auto KV: beforeBBName2BB){
            root["beforeBBName2BB"][KV.first] = KV.second;
        }
        map<string, string> afterBBName2BB = P.getAfterBBName2BB();
        for(auto KV: afterBBName2BB){
            root["afterBBName2StartWith"][KV.first] = P.getTheLastInstructionName(KV.first, P.getAfterFunction());
            root["afterBBName2BB"][KV.first] = KV.second;
            root["afterBBName2LeastBeforeScope"][KV.first] = P.getLeastBeforeBBsofTargetBB(KV.first);
            root["afterBBName2MostBeforeScope"][KV.first] = P.getMostBeforeBBsofTargetBB(KV.first);
            // if for vec, provide the corresponding src of the BB; provide the src of BB in
            if(forVec){
                root["afterBBLoc"][KV.first] = P.getLocOfBB(KV.first);
            }
        }
        map<string, string> afterBBName2BBIn = P.getAfterBBName2BBIn();
        for(auto KV: afterBBName2BBIn){
            root["afterBBName2BBIn"][KV.first] = KV.second;
        }
        map<string, string> afterBBName2BBOut = P.getAfterBBName2BBOut();
        for(auto KV: afterBBName2BBOut){
            root["afterBBName2BBOut"][KV.first] = KV.second;
        }

        map<string, string> afterBB2OriginInsts = P.getAfterBB2OriginInst();
        for(auto KV: afterBB2OriginInsts){
            root["afterBBName2OriginI"][KV.first] = KV.second;
        }
        root["sourceFileWithFuncName"] = stringSplit(P.getBeforeFunction()->getParent()->getSourceFileName(), '/').back()+"."+P.getBeforeFunction()->getName().str()+".log";

        writer->write(root, &outfile);
        outfile<<"\n";
    }
}

// Prompt contains origin cfg, the cfg after split, the IR for each BB
void genUnopt2OptWithSmallerBBForPrompt(string InputFilePath, string OutputFilePath, int InstructionThreshold){
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
            map<string, Function*> beforeFuncName2FuncPtr = getFuncName2FuncPtr(beforeModuleIR);
            map<string, Function*> afterFuncName2FuncPtr = getFuncName2FuncPtr(afterModuleIR);
            vector<PromptWithSmallerBBInfo> prompts = getPrompts(beforeFuncName2FuncPtr, afterFuncName2FuncPtr, InstructionThreshold);
            writeIntoJson(prompts, outfile);

            // free
            for(PromptWithSmallerBBInfo &p: prompts){
                if(p.getBeforeRootScope())
                    delete p.getBeforeRootScope();
                if(p.getAfterRootScope())
                    delete p.getAfterRootScope();
            }
        }
    }

    infile.close();
    outfile.close();

}

map<string, Function*> getLoopFuncName2FuncPtr(map<string, Function*> FuncName2FuncPtr){
    string keyword = ".qLOOP.";
    map<string, Function*> res;
    for(auto &KV: FuncName2FuncPtr){
        string funcName = KV.first;
        if(funcName.find(keyword) != string::npos)
            res[funcName] = KV.second;
    }
    return res;
}

// Prompt contains origin cfg, the cfg after split, the IR for each BB
void genUnopt2OptWithSmallerBBForVecPrompt(string InputFilePath, string OutputFilePath, int InstructionThreshold){
    forVec = true;

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

            map<string, Function*> beforeFuncName2FuncPtr = getLoopFuncName2FuncPtr(getFuncName2FuncPtr(beforeModuleIR));
            map<string, Function*> afterFuncName2FuncPtr = getLoopFuncName2FuncPtr(getFuncName2FuncPtr(afterModuleIR));
            
            vector<PromptWithSmallerBBInfo> prompts = getPrompts(beforeFuncName2FuncPtr, afterFuncName2FuncPtr, InstructionThreshold);
            writeIntoJson(prompts, outfile);

            // free
            for(PromptWithSmallerBBInfo &p: prompts){
                if(p.getBeforeRootScope())
                    delete p.getBeforeRootScope();
                if(p.getAfterRootScope())
                    delete p.getAfterRootScope();
            }
        }
    }

    infile.close();
    outfile.close();

}