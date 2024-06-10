#ifndef __GEN_MODEL_TRAIN_DATA__
#define __GEN_MODEL_TRAIN_DATA__

#include <vector>
#include <string>

/**
 * @brief generate the first step train data, and need to be normalized(?) in the fulture(we can try donnot normalize). The Input is json file, that contains 2 columns, one is Module IR before optimize, one is Module IR after optimize.
*/
void genRawTrainDataForPrompt(std::string InputFilePath, std::string OutputFilePath);
#endif