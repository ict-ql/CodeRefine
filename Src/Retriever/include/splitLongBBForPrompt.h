#ifndef __SPLIT_LONG_BB__
#define __SPLIT_LONG_BB__

#include <string>
// Prompt contains origin cfg, the cfg after split, the IR for each BB
// When a BB containing Instruction Num over InstructionThreshold, split it
void genUnopt2OptWithSmallerBBForPrompt(std::string InputFilePath, std::string OutputFilePath, int InstructionThreshold);
void genUnopt2OptWithSmallerBBForVecPrompt(std::string InputFilePath, std::string OutputFilePath, int InstructionThreshold);

#endif