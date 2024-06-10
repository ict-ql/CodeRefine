#ifndef __UTILS__
#define __UTILS__

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

#define DIM 300
using Vector = llvm::SmallVector<double, DIM>;

std::unique_ptr<llvm::Module> getLLVMIR(std::string IRStr);
std::unique_ptr<llvm::Module> getLLVMIRFromFile(std::string filepath);
void writeRecordsIntoJson(std::vector<std::map<std::string, std::string>> &records, 
                          std::ofstream &outfile);
std::vector<std::string> getAllFiles(const std::string& directory);
std::vector<std::string> stringSplit(const std::string& str, char delim);
std::string getVectorAsStr(Vector vec);
#endif