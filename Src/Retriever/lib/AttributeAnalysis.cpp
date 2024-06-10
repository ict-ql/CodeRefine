#include "utils.h"
#include "AttributeAnalysis.h"

#include <iostream>
#include <fstream>
#include <set>

using namespace std;

class CustomedCallGraphNode{
    const Function * F;
    // bool isCycle; // the CallGraphNode has occured in the graph before
    list<CustomedCallGraphNode*> children;
public:
    static map<const Function*, CustomedCallGraphNode*> Function2CallGraphNode;

    CustomedCallGraphNode(const Function * F) : F(F), children({}){}
    // ~CustomedCallGraphNode(){
    //     for(CustomedCallGraphNode *c: children){
    //         delete c;
    //     }
    // }
    const Function * getFunction(){
        return F;
    }
    list<CustomedCallGraphNode*> getChildren(){
        return children;
    }
    void addChild(CustomedCallGraphNode *child){
        children.push_back(child);
    }
    static void dump(CustomedCallGraphNode *node, std::set<CustomedCallGraphNode*> hasDumped={}, string prefix="|"){
        errs() << prefix << node->getFunction()->getName()<<"\n";
        string newPrefix = prefix + "\t|";
        if(hasDumped.find(node) != hasDumped.end())
            return;
        hasDumped.insert(node);
        for(auto c: node->getChildren()){
            CustomedCallGraphNode::dump(c, hasDumped, newPrefix);
        }
    }
    // void calcPathsToSpecificFunctionName(string specificFunctionName);
};

map<const Function*, CustomedCallGraphNode*> CustomedCallGraphNode::Function2CallGraphNode = {};

/**
 * @brief use bfs to search all path from current to the node with specific function name
*/
// void CustomedCallGraphNode::calcPathsToSpecificFunctionName(string specificFunctionName){
//     list<list<CustomedCallGraphNode*>> res;
//     set<CustomedCallGraphNode*> visited;
//     queue<list<CustomedCallGraphNode*>> q;
//     q.push({this});
//     visited.insert(this);

//     while(!q.empty()){
//         int size = q.size();
//         for (int i=0; i<size; i++){
//             list<CustomedCallGraphNode*> path = q.front();
//             q.pop();

//             CustomedCallGraphNode* cur = path.back();
//             if(cur->getFunction()->getName().str() == specificFunctionName){
//                 res.push_back(path);
//             } else{
//                 for ()
//             }
//         }
//     }
// }


static bool ifStrListContainsStr(list<string> strList, std::string strr){
  if(find(strList.begin(), strList.end(), strr) != strList.end()){
    return true;
  }
  return false;
}

// static map<string, int> BasicAPI2OccureNum;


CustomedCallGraphNode* getCompleteCallGraphDetail(CallGraphNode *node, list<string> basicAPIs,
                                                    set<string> &OccureAPIs){
    if(!node->getFunction())
        return NULL;
    if(CustomedCallGraphNode::Function2CallGraphNode.find(node->getFunction())
        != CustomedCallGraphNode::Function2CallGraphNode.end()){
            if(ifStrListContainsStr(basicAPIs, node->getFunction()->getName().str()))
                OccureAPIs.insert(node->getFunction()->getName().str());
            //     BasicAPI2OccureNum[*find(basicAPIs.begin(), basicAPIs.end(), node->getFunction()->getName())] += 1;
        return CustomedCallGraphNode::Function2CallGraphNode[node->getFunction()];
    }
    
    CustomedCallGraphNode* cNode = new CustomedCallGraphNode(node->getFunction());
    CustomedCallGraphNode::Function2CallGraphNode[node->getFunction()] = cNode;
    if(!ifStrListContainsStr(basicAPIs, node->getFunction()->getName().str())){
        for(auto itr=node->begin(), ie=node->end(); itr!=ie; itr++){
            CustomedCallGraphNode* child = getCompleteCallGraphDetail(itr->second, basicAPIs, OccureAPIs);
            if(child)
                cNode->addChild(child);
        }
    }
    else{
        // BasicAPI2OccureNum[*find(basicAPIs.begin(), basicAPIs.end(), node->getFunction()->getName())] += 1;
        OccureAPIs.insert(node->getFunction()->getName().str());
    }
    return cNode;
}

PreservedAnalyses AttributeAnalysisPass::run(Module &M, ModuleAnalysisManager &AM){
    list<CustomedCallGraphNode*> focusCallGraphs;
    map<string, int> BasicAPI2OccureNum;

    // Step1: create Call graph and get the call graph 
    CallGraph CG(M);
    for(auto itr=CG.begin(), ie=CG.end(); itr!=ie; itr++){
        if(itr->first 
            && ifStrListContainsStr(rootFuncNames, itr->first->getName().str())){
            set<string> OccureAPIs;
            focusCallGraphs.push_back(getCompleteCallGraphDetail(itr->second.get(),
                                                                    basicAPIs,
                                                                    OccureAPIs));
            for(string s: OccureAPIs){
                BasicAPI2OccureNum[s] += 1;
            }
        }
    }
    
    // for(CustomedCallGraphNode* cNode: focusCallGraphs){
    //     CustomedCallGraphNode::dump(cNode);
    //     errs()<<"\n";
    // }
    for(auto KV: BasicAPI2OccureNum){
        errs()<<KV.first<<"\t"<<KV.second<<"\n";
    }

    // StepN: delete
    for(auto &KV: CustomedCallGraphNode::Function2CallGraphNode){
        delete KV.second;
    }
    return PreservedAnalyses::all();
}

list<string> readFileAsList(string FilePath){
  list<string> lst;
  ifstream fin;
  if(FilePath == "")
    return lst;
  fin.open(FilePath);
  if (!fin) {
    errs() << "Cannot open the file.\n";
    return lst;
  }
  string line;
  while (getline(fin, line)) {
    lst.push_back(line);
  }
  fin.close();
  return lst;
}

void runAttributeAnalysis(string InputFilePath, string FocusFuncFilePath, 
                            string BasicAPIFilePath, string OutputFilePath){
    list<string> rootFuncNames = readFileAsList(FocusFuncFilePath);
    list<string> basicAPIs = readFileAsList(BasicAPIFilePath);
    // create opt pipeline
    ModulePassManager PM;
    ModuleAnalysisManager AM;
    PM.addPass(AttributeAnalysisPass(rootFuncNames, basicAPIs));

    std::unique_ptr<Module> M = getLLVMIRFromFile(InputFilePath);
    PM.run(*M, AM);
}