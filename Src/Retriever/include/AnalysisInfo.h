#ifndef __ANALYSIS_INFO__
#define __ANALYSIS_INFO__

#include <string>
#include <map>

struct AnalysisInfo{
    std::string FuncIR;
    std::string BlockFrequencyAnalysisResult;
    std::string DemandedBitsAnalysisResult;
    std::string DominatorTreeAnalysisResult;
    std::string LoopInfoResult;
    std::string LoopAccessAnalysisResult;
    std::string MemorySSAAnalysisResult;
    std::string ScalarEvolutionAnalysisResult;
    std::string CFGRelatedResult;
    std::string DFRelatedResult;
    std::string AAResult;
    std::string InstResult;
    std::string DataLayoutResult;
    std::string GlobalVarResult;
    std::string DeclareFuncResult;

    std::map<std::string, std::string> getNoEmptyFields(){
        std::map<std::string, std::string> res;
        if(FuncIR != "")
            res["FuncIR"] = FuncIR;
        if(BlockFrequencyAnalysisResult != "")
            res["BlockFrequencyAnalysisResult"] = BlockFrequencyAnalysisResult;
        if(DemandedBitsAnalysisResult != "")
            res["DemandedBitsAnalysisResult"] = DemandedBitsAnalysisResult;
        if(DominatorTreeAnalysisResult != "")
            res["DominatorTreeAnalysisResult"] = DominatorTreeAnalysisResult;
        if(LoopInfoResult != "")
            res["LoopInfoResult"] = LoopInfoResult;
        if(LoopAccessAnalysisResult != "")
            res["LoopAccessAnalysisResult"] = LoopAccessAnalysisResult;
        if(MemorySSAAnalysisResult != "")
            res["MemorySSAAnalysisResult"] = MemorySSAAnalysisResult;
        if(ScalarEvolutionAnalysisResult != "")
            res["ScalarEvolutionAnalysisResult"] = ScalarEvolutionAnalysisResult;
        if(CFGRelatedResult != "")
            res["CFGRelatedResult"] = CFGRelatedResult;
        if(DFRelatedResult != "")
            res["DFRelatedResult"] = DFRelatedResult;
        if(AAResult != "")
            res["AAResult"] = AAResult;
        if(InstResult != "")
            res["InstResult"] = InstResult;
        if(DataLayoutResult != "")
            res["DataLayoutResult"] = DataLayoutResult;
        if(GlobalVarResult != "")
            res["GlobalVarResult"] = GlobalVarResult;
        if(DeclareFuncResult != "")
            res["DeclareFuncResult"] = DeclareFuncResult;
        return res;
    }
};

#endif