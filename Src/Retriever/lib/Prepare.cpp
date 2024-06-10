#include "Prepare.h"
#include <fstream>

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "utils.h"

using namespace std;
using namespace llvm;

void getChangeIRFunc(string InputFilePath, string OutputFilePath){
    auto M = getLLVMIRFromFile(InputFilePath);

    ofstream outfile;
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");
    
    bool containMemoryWrite = false;
    for(Function &F: *M){
        if(F.isDeclaration())
            continue;
        containMemoryWrite = false;
        for(BasicBlock &BB: F){
            for(Instruction &I: BB){
                if(isa<StoreInst>(&I)){
                    containMemoryWrite = true;
                    break;
                }
            }
            if(containMemoryWrite)
                break;
        }
        if(containMemoryWrite)
            outfile<<F.getName().str()<<"\n";
    }
    outfile.close();
}