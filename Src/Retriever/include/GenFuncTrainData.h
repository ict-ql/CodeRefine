#ifndef __GEN_FUNC_TRAIN_DATA__
#define __GEN_FUNC_TRAIN_DATA__

#include <vector>
#include <string>

void genFunctionTrainData(std::vector<std::string> InputFilePaths, std::string OutputFilePath);

void processSplitPassData(std::vector<std::string> InputFilePaths, std::string OutputFilePath);

void genFunctionTrainData(std::string InputFilePath, std::string OutputFilePath);

void genJustFuncIRFromModuleIR(std::string InputFilePath, std::string OutputFilePath);

#endif