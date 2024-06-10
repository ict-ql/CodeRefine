#include "FilterRegionSimilarToTarget.h"
#include "IR2Vec.h"
#include "utils.h"

#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Instructions.h"
#include <fstream>
#include <json/json.h>
#include <cmath>

using namespace std;
using namespace llvm;

Vector convertStrToVector(string vectorStr){
    Vector rep;
    vector<string> vec = stringSplit(vectorStr, ',');
    for(string val: vec){
        if(val != ""){
            rep.push_back(stod(val));
        }
    }
    return rep;
}

void getLoopVec(string TargetVecPath, std::map<std::string, Vector> &loopName2VecMap){
    Json::Reader reader;
    Json::Value root;
    string line;
    ifstream infile(TargetVecPath);
    assert(infile.is_open() && "ERROR: open input file failed!\n");
    while(getline(infile, line)){
        if(reader.parse(line, root)){
            loopName2VecMap[root["FuncName"].asString()] = convertStrToVector(root["loopVec"].asString());
        }
    }
}

// Euclidean Distance
static double calcDistance(Vector v1, Vector v2){
    Vector funcVector(DIM, 0);
    std::transform(v1.begin(), v1.end(), v2.begin(), funcVector.begin(), std::minus<double>());

    // double (*fabs)(double) = &std::abs;
    // std::transform(funcVector.begin(), funcVector.end(), funcVector.begin(), fabs);

    double sum = 0;
    for (double d: funcVector){
        sum += d*d;
    }
    return std::sqrt(sum)/DIM;
}

/**
 * @brief update the Gen and Kill set according to the Instruction `I`
 * @return std::set<Value*> &Gen, std::set<Value*> &Kill
*/
static void updateGenKill(Instruction *I, set<Instruction*> &Gen, 
                            set<Instruction*> &Kill){
    if(I->getNumUses() != 0){
        Kill.insert(I);
    }

    for(auto opItr = I->op_begin(); opItr != I->op_end(); ++opItr){ // add the operand to Gen
        if(Instruction *I = dyn_cast<Instruction>(*opItr))
            Gen.insert(I);
    }
}

/**
 * @brief calculate the Loop's Gen and Kill 
 * @return std::set<Value*> &Gen, std::set<Value*> &Kill
*/
static void getLoopGenKill(const Loop *L, set<Instruction*> &Gen, 
                            set<Instruction*> &Kill){
    for(auto BB: L->getBlocks()){
        for(auto &I: *BB){
            updateGenKill(&I, Gen, Kill);
        }
    }
}

/**
 * @brief calculate the liveness variable outside the loop
 * @return return the liveness variable outside the loop
*/
static void calculateLivenessVariable(const Loop* L, set<Instruction*> &LV){
    set<BasicBlock*> BBs;
    for(auto BB: L->getBlocks()){
        BBs.insert(BB);
    }

    // this instruction is used by instructions that doesn't in the loop
    for(auto BB: BBs){
        for(auto &I: *BB){
            for(auto v: I.users()){
                if(Instruction *tmpI = dyn_cast<Instruction>(v)){
                    if(BBs.find(tmpI->getParent()) == BBs.end()){
                        LV.insert(&I);
                        break;
                    }
                }
            }
        }
    }
}

static void copyBBToNewF(BasicBlock *BB, Function *newF, ValueToValueMapTy &VMap){
    LLVMContext &Ctx = newF->getContext();
    BasicBlock *newBB = BasicBlock::Create(Ctx, BB->getName(), newF);
    VMap[BB] = newBB;
    for(auto &I: *BB){
        Instruction *newI = I.clone();
        newI->setName(I.getName());
        VMap[&I] = newI;
        newI->insertInto(newBB, newBB->end());
        RemapInstruction(newI, VMap,
            llvm::RF_NoModuleLevelChanges | llvm::RF_IgnoreMissingLocals);
    }
}

bool skipTheLoop(const Loop* L){
    for(auto BB: L->getBlocks()){
        for(auto &I: *BB){
            if(isa<ReturnInst>(I))
                return true;
            if(isa<InvokeInst>(I))
                return true;
            // if(isa<CallInst>(I))
            //     return true;
        }
    }
    return false;
}

static void createNewFunctionForSpecificLoops(set<const Loop*> Loops, Function *originF){
    // step0: calc each loop's In and Out
    // errs()<<"--------------------\n";

    // step0.1: in
    map<const Loop*, set<Instruction*>> L2In;
    for(auto L: Loops){
        std::set<Instruction*> Gen;
        std::set<Instruction*> Kill;
        getLoopGenKill(L, Gen, Kill);
        set_difference(Gen.begin(), Gen.end(),
                        Kill.begin(), Kill.end(),
                        std::inserter(L2In[L], L2In[L].end()));
    }

    // step0.2: out
    map<const Loop*, set<Instruction*>> L2Out;
    for(auto L: Loops){
        calculateLivenessVariable(L, L2Out[L]);
    }

    // for(auto L: Loops){
    //     L->print(errs());
    //     errs()<<"\n";
    //     for(auto II: L2Out[L]){
    //         II->printAsOperand(errs());
    //         errs()<<", ";
    //     }
    //     errs()<<"\n";
    // }

    int cnt = 0;
    for(const Loop *Lptr: Loops){
        if(skipTheLoop(Lptr))
            continue;
        set<Instruction*> In = L2In[Lptr];
        set<Instruction*> Out = L2Out[Lptr];
        std::vector<Type*> paramsTypes;
        for(Argument &arg: originF->args()){
            paramsTypes.push_back(arg.getType());
        }
        for(Instruction *I: In){
            // llvm::errs()<<"Func Arg:";I->print(llvm::errs()); llvm::errs()<<"\n";
            paramsTypes.push_back(I->getType());
        }
        for(Instruction *I: Out){
            // I->print(llvm::errs()); llvm::errs()<<"\n";
            paramsTypes.push_back(PointerType::get(I->getType(), 0));
        }
        Function *newF = llvm::Function::Create(FunctionType::get(Type::getVoidTy(originF->getContext()), 
                                                                    paramsTypes, false), 
                                                llvm::GlobalValue::LinkageTypes::ExternalLinkage,
                                                originF->getName()+".qLOOP."+std::to_string(cnt));
        originF->getParent()->getFunctionList().insert(originF->getIterator(), newF); // insert to Moddule
        
        LLVMContext &Ctx = newF->getContext();
        BasicBlock *entryBB = BasicBlock::Create(Ctx, "", newF); // create entry BB

        try{
            // step2: map the `In` of the region to the Function Argument
            ValueToValueMapTy VMap;
            assert(originF->arg_size()+In.size()+Out.size() == newF->arg_size() && "The new Function's argument must contain the Region's In and Out!\n");
            int arg_num = 0;
            for(Argument &arg: originF->args()){
                VMap[&arg] = newF->getArg(arg_num);
                arg_num++;
            }

            for(Instruction *I: In){
                VMap[I] = newF->getArg(arg_num);
                arg_num++;
            }

            std::map<Instruction*, Value*> OutOriginI2OutArg;
            for(Instruction *I: Out){
                OutOriginI2OutArg[I] = newF->getArg(arg_num);
                arg_num++;
            }

            // step3: insert all instruction to the new one
            for(auto BB: Lptr->getBlocks()){
                copyBBToNewF(BB, newF, VMap);
            }

            // step4: create Exit Blocks at the end
            SmallVector<BasicBlock*> exitBBs;
            Lptr->getExitBlocks(exitBBs);
            for(auto BB: exitBBs){
                BasicBlock *newBB = BasicBlock::Create(Ctx, BB->getName(), newF);
                VMap[BB] = newBB;
            }
            // map the entryBB to the loop's preheader
            if(Lptr->getLoopPreheader())
                VMap[Lptr->getLoopPreheader()] = entryBB;


            // step5: remap at last(remap BB name)
            for(auto &newBB: *newF){
                for(auto &newI: newBB){
                    RemapInstruction(&newI, VMap,
                        llvm::RF_NoModuleLevelChanges | llvm::RF_IgnoreMissingLocals);
                }
            }

            // step6: add a br instr to jump to loop header
            BasicBlock *newLoopHeader = dyn_cast<BasicBlock>(VMap.lookup(Lptr->getHeader()));
            assert(newLoopHeader && "The Out Origin BB must have been inserted to the new Function before!\n");
            BranchInst::Create(newLoopHeader, entryBB);

            // step7: insert store result to the out argument to the exit block
            for(auto &BB: *newF){
                if (BB.getTerminator())
                    continue;
                DominatorTree DT(*newF);
                for(auto &KV: OutOriginI2OutArg){
                    Instruction *I = KV.first;
                    Value *arg = KV.second;
                    Instruction *newI = dyn_cast<Instruction>(VMap.lookup(I));
                    // if(!newI){
                    //     I->print(errs());
                    //     errs()<<" cannot be found\nNew Func:\n";
                    //     newF->print(errs());
                    //     errs()<<"\ncorresponding to LOOP:\n";
                    //     Lptr->print(errs());
                    //     errs()<<"\ncorresponding to Function:\n";
                    //     originF->print(errs());
                    // }
                    
                    assert(newI && "The Out Origin Instruction must have been inserted to the new Function before!\n");

                    if (DT.dominates(newI, &BB)){
                        new StoreInst(newI, arg, false, Align(), &BB);
                    }                
                }
                ReturnInst::Create(Ctx, &BB); // the new Function Type is `void`
            }

            // // llvm::errs()<<"===============F===============\n";
            // // newF->print(llvm::errs()); llvm::errs()<<"\n";
            cnt++;

            // newF->print(errs());
            assert(!verifyFunction(*newF, &errs()));
        }catch(const char* msg){
            errs()<<msg<<"\n";
            originF->getParent()->getFunctionList().erase(newF);;
        }
    }
}

static void preprocessFile(string filePath, string VocabPath, string OutputDir,
                            double SimilarityThreshold, 
                            const std::map<std::string, Vector> &loopName2VecMap){
    unique_ptr<Module> M;
    try{
        M = getLLVMIRFromFile(filePath);
    }catch(const char* msg){
        errs()<<msg<<"\n";
        return;
    }

    map<Function*, LoopInfo*> func2LI;
    vector<DominatorTree *> DTs;
    for(auto &F: *M){
        if(F.isDeclaration())
            continue;
        DominatorTree *DT = new DominatorTree(F);
        DTs.push_back(DT);
        func2LI[&F] = new LoopInfo(*DT);
    }
    
    

    // calculate Each loop's vector
    auto ir2vec = IR2Vec::Embeddings(*M, func2LI,
                                        IR2Vec::IR2VecMode::FlowAware,
                                        VocabPath, 'r');
    auto loopVecMap = ir2vec.getLoopVecMap();
    if(loopVecMap.size() == 0)
        return;
    
    // for(auto &KV: loopVecMap){
    //     auto *L = KV.first;
    //     Vector v = KV.second;
    //     errs()<<"----------------\nFunc Name: "<<L->getHeader()->getParent()->getName().str()<<":";
    //     L->print(errs());
    //     errs()<<"\n"<<getVectorAsStr(v)<<"\n";
    //     // break;
    // }

    // get the loop that similar to target
    map<Function*, set<const Loop*>> func2loops;
    for(auto &KV: loopVecMap){
        auto *L = KV.first;
        Function *F = L->getHeader()->getParent();
        Vector v = KV.second;
        for(auto KV2: loopName2VecMap){
            double dist = calcDistance(v, KV2.second);
            if(dist < SimilarityThreshold){
                if(func2loops.find(F) != func2loops.end())
                    func2loops[F].insert(L);
                else{
                    func2loops[F] = {L};
                }
            }
        }
    }

    // create new function for these loops
    for(auto &KV: func2loops){
        createNewFunctionForSpecificLoops(KV.second, KV.first);
    }

    // print new IR into file
    
    ofstream outfile;
    string OutputFilePath = OutputDir+"/"+stringSplit(M->getSourceFileName(), '/').back()+".ll";
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");
    string MIRStr = "";
    raw_string_ostream MIROS(MIRStr);
    M->print(MIROS, nullptr);
    outfile<<MIRStr;
    outfile.close();

    // create a new Module/Function?, contain the new loop

    // calculate the distance of loop vector with target vector
    // map<const Loop*, map<string, double>> loop2DistanceMap;
    // for(auto &KV: loopVecMap){
    //     auto *L = KV.first;
    //     Vector v = KV.second;
    //     for(auto KV2: loopName2VecMap){
    //         double dist = calcDistance(v, KV2.second);
    //         loop2DistanceMap[L][KV2.first] = dist;
    //     }
    // }
    // for(auto &KV: loop2DistanceMap){
    //     auto *L = KV.first;
    //     errs()<<"-------"<<L->getHeader()->getParent()->getName()<<"------\n";
        
    //     L->print(errs());
    //     for(auto &KV2: KV.second){
    //         errs()<<"\n"<<KV2.first<<"\t"<<KV2.second;
    //     }
    // }

    // free loop info/DTs
    for(auto &KV: func2LI){
      delete KV.second;
    }
    for(auto &DT: DTs){
      delete DT;
    }
}

void filterLoopAccordingToSimilarity(string InputFileDir, string TargetVecPath, 
                                        string VocabPath, string OutputDir, 
                                        double SimilarityThreshold){
    std::map<std::string, Vector> loopName2VecMap;
    getLoopVec(TargetVecPath, loopName2VecMap);
    // for(auto &KV: loopName2VecMap){
    //     errs()<<KV.first<<"\t"<<getVectorAsStr(KV.second)<<"\n";
    //     break;
    // }

    vector<string> inputIrPaths = getAllFiles(InputFileDir);
    assert(inputIrPaths.size() != 0 && "ERROR: the InputFileDir contains none files!\n");

    for(string path: inputIrPaths){
        preprocessFile(path, VocabPath, OutputDir, SimilarityThreshold, loopName2VecMap);
        // break;
    }
}