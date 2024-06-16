add_executable(TOOL Main.cpp GenFuncTrainData.cpp GenAnalysisTrainData.cpp CollectModuleAnalysisInfo.cpp GenEachFuncStatusAfterPass.cpp Prepare.cpp AttributeAnalysis.cpp GenActionModelTrainData.cpp BBLevelUnopt2Opt.cpp splitLongBBForPrompt.cpp FilterRegionSimilarToTarget.cpp Scope.cpp utils.cpp json/jsoncpp.cpp)
llvm_map_components_to_libnames(llvm_libs support core irreader analysis TransformUtils Passes)
target_link_libraries (TOOL ${llvm_libs}  "IR2Vec/build/lib/libIR2Vec.so")
