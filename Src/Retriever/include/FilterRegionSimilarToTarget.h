#ifndef __FILTER_REGION__
#define __FILTER_REGION__

#include <string>

void filterLoopAccordingToSimilarity(std::string InputFileDir, std::string TargetVecPath, 
                                        std::string VocabPath, std::string OutputDir, 
                                        double SimilarityThreshold);

#endif