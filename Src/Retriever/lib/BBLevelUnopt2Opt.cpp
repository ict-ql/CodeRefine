#include "BBLevelUnopt2Opt.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <set>
#include <list>
#include <algorithm>
#include <string>
#include <json/json.h>

#include "llvm/IR/InstIterator.h"

using namespace std;
using namespace llvm;

class PromptRegion{
    set<unsigned> lines; // the lines in source code of the BBs in this region
    set<BasicBlock*> BBs;
public:
    PromptRegion(): lines({}), BBs({}){}
    PromptRegion(BasicBlock* BB);
    PromptRegion(const PromptRegion& r1, const PromptRegion& r2); // create new region from original region
    
    static bool has_intersection(const PromptRegion& r1, const PromptRegion& r2);

    set<BasicBlock*> getBBs() const { return BBs; }
    set<unsigned> getLines() const { return lines; }

    void addBBToRegion(BasicBlock* BB); // add the element(BB and lines)

    string toString(){
        string ans = "";
        raw_string_ostream OS(ans);
        // OS << "line: {";
        // for(unsigned line: lines){
        //     OS << to_string(line) + ", ";
        // }
        // OS << "} BB: {";
        // for(auto BB: BBs){
        //     BB->printAsOperand(OS);
        //     OS << ", ";
        // }
        for(auto BB: BBs){
            BB->print(OS);
        }
        // OS << "}";
        return ans;
    }

    Function* getParent(){
        if(BBs.size() == 0)
            return NULL;
        return (*BBs.begin())->getParent();
    }
};

class Prompt{
    PromptRegion beforeOptRegion;
    PromptRegion attributes;
    PromptRegion afterOptRegion;
public:
    Prompt() {}
    Prompt(PromptRegion beforeOptRegion, PromptRegion attributes, PromptRegion afterOptRegion)
            : beforeOptRegion(beforeOptRegion), attributes(attributes), afterOptRegion(afterOptRegion) {}
    
    string toString(){
        string ans = "";
        raw_string_ostream OS(ans);
        OS<<"beforeOptRegion:\n";
        OS<<beforeOptRegion.toString()<<"\n";
        OS<<"attributes:\n";
        OS<<attributes.toString()<<"\n";
        OS<<"afterOptRegion:\n";
        OS<<afterOptRegion.toString()<<"\n";
        return ans;
    }

    // get the unopt region(input)
    string getBeforeOptRegionAsString(){
        return beforeOptRegion.toString();
    }

    // get the attributes(input)
    string getAttributsAsString(){
        return attributes.toString();
    }

    // get the opt region(output)
    string getAfterOptRegionAsString(){
        return afterOptRegion.toString();
    }

    Function* getUnoptFunction(){
        return beforeOptRegion.getParent();
    }

    Function* getOptFunction(){
        return afterOptRegion.getParent();
    }

    string getUnoptFunctionAsString(){
        string ans = "";
        raw_string_ostream OS(ans);
        if(getUnoptFunction())
            getUnoptFunction()->print(OS);
        return ans;
    }

    string getOptFunctionAsString(){
        string ans = "";
        raw_string_ostream OS(ans);
        if(getOptFunction())
            getOptFunction()->print(OS);
        return ans;
    }
};

PromptRegion::PromptRegion(BasicBlock* BB){
    lines.clear();
    BBs.clear();
    addBBToRegion(BB);
}

PromptRegion::PromptRegion(const PromptRegion& r1, const PromptRegion& r2){
    const set<unsigned> r1Lines = r1.getLines();
    const set<unsigned> r2Lines = r2.getLines();
    set_union(r1Lines.begin(), r1Lines.end(), 
                r2Lines.begin(), r2Lines.end(), 
                inserter(lines, lines.begin()));
    
    set<BasicBlock*> r1BBs = r1.getBBs();
    set<BasicBlock*> r2BBs = r2.getBBs();
    set_union(r1BBs.begin(), r1BBs.end(), 
                r2BBs.begin(), r2BBs.end(), 
                inserter(BBs, BBs.begin()));
}

bool PromptRegion::has_intersection(const PromptRegion& r1, const PromptRegion& r2){
    set<unsigned> result;
    const set<unsigned> r1Lines = r1.getLines();
    const set<unsigned> r2Lines = r2.getLines();
    set_intersection(r1Lines.begin(), r1Lines.end(), 
                        r2Lines.begin(), r2Lines.end(), 
                        inserter(result, result.begin()));
    return !result.empty();
}

void PromptRegion::addBBToRegion(BasicBlock* BB){
    BBs.insert(BB);
    for(Instruction &I: *BB){
        if(I.getDebugLoc()){
            lines.insert(I.getDebugLoc().getLine());
        }
    }
}

void merge_regions(list<PromptRegion> &lst){
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it1 = lst.begin(); it1 != lst.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != lst.end(); ++it2) {
                if (PromptRegion::has_intersection(*it1, *it2)){
                    lst.push_back(PromptRegion(*it1, *it2)); // add new one
                    lst.erase(it1); // delete old one
                    lst.erase(it2);
                    changed = true;
                    break;
                }
            }
            if (changed) break; 
        }
    }
}

void gen_regions_corresponding_from_src_regions(const list<PromptRegion> &src_regions,
                                                unique_ptr<Module> &M,
                                                list<Prompt> &res_prompts_info){
    // Step0: init, create the line num to BBs map
    map<unsigned, set<BasicBlock*>> line2BBs;
    for(Function &F: *M){
        for (auto &I : instructions(F)) {
            if(I.getDebugLoc()){
                line2BBs[I.getDebugLoc().getLine()].insert(I.getParent());
            }
        }
    }
    map<unsigned, set<BasicBlock*>> line2unoptBBs;
    for(auto &r: src_regions){
        set<BasicBlock*> BBs = r.getBBs();
        for(BasicBlock* BB: BBs){
            for(auto &I : *BB){
                if(I.getDebugLoc()){
                    line2unoptBBs[I.getDebugLoc().getLine()].insert(I.getParent());
                }
            }
        }
    }

    // Step1: create Prompt
    for(auto &r: src_regions){
        // create after opt region
        const set<unsigned> lines = r.getLines();
        PromptRegion optRegion;
        for(unsigned line: lines){
            for(BasicBlock* BB: line2BBs[line])
                optRegion.addBBToRegion(BB);
        }

        // create attributs
        PromptRegion attributes;
        const set<unsigned> optLines = optRegion.getLines();
        set<unsigned> resLines;
        set_difference(optLines.begin(), optLines.end(),
                        lines.begin(), lines.end(),
                        inserter(resLines, resLines.begin()));
        for(unsigned line: resLines){
            if(line2unoptBBs.count(line)){
                for(BasicBlock* BB: line2unoptBBs[line])
                    attributes.addBBToRegion(BB);
            }
        }
        res_prompts_info.push_back(Prompt(r, attributes, optRegion));
    }
}

void gen_regions_through_line_num(unique_ptr<Module> &M, list<PromptRegion> &regions){
    // Step0: init, each basic block is a region
    for(Function &F: *M){
        for(BasicBlock &BB: F){
            regions.push_back(PromptRegion(&BB));
        }
    }
    // errs()<<"Init:\n";
    // for(auto &r: regions){
    //     errs()<<r.toString()<<"\n";
    // }

    // Step1: merge
    merge_regions(regions);
    // errs()<<"After merge:\n";
    // for(auto &r: regions){
    //     errs()<<r.toString()<<"\n";
    // }
}

void genUnoptBB2OptForPrompt(string InputFilePath, string OutputFilePath){
    errs()<<"WARNING!: all ir must be compile by `-g`!\n";

    Json::Reader reader;
    Json::Value root;
    ifstream infile(InputFilePath);
    assert(infile.is_open() && "ERROR: open input file failed!\n");
    ofstream outfile;
    outfile.open(OutputFilePath, std::ios::out | std::ios::trunc);
    assert(outfile.is_open() && "ERROR: open output file failed!\n");

    string line;
    int moduleIdx = 0;
    while(getline(infile, line)){
        moduleIdx += 1;
        if(reader.parse(line, root)){
            string beforeModuleIRstr = root["before_opt_module_ir"].asString();
            string afterModuleIRstr = root["after_opt_module_ir"].asString();
            unique_ptr<Module> beforeModuleIR;
            unique_ptr<Module> afterModuleIR;
            try{
                beforeModuleIR = getLLVMIR(beforeModuleIRstr);
                afterModuleIR = getLLVMIR(afterModuleIRstr);
            }catch(const char* msg){
                errs()<<msg<<"\n";
                continue;
            }
            assert(beforeModuleIR.get() && afterModuleIR.get() && "ERROR: must not empty");
            list<PromptRegion> regions;
            list<Prompt> res_prompts_info;
            gen_regions_through_line_num(beforeModuleIR, regions);
            // get the optimized
            gen_regions_corresponding_from_src_regions(regions,
                                                        afterModuleIR,
                                                        res_prompts_info);
            
            // for(auto &p: res_prompts_info){
            //     errs()<<p.toString()<<"\n";
            // }

            vector<map<string, string>> records;
            for(auto &p: res_prompts_info){
                map<string, string> record;
                record["beforeOptRegion"] = p.getBeforeOptRegionAsString();
                record["attributes"] = p.getAttributsAsString();
                record["afterOptRegion"] = p.getAfterOptRegionAsString();
                record["unoptFunc"] = p.getUnoptFunctionAsString();
                record["optFunc"] = p.getOptFunctionAsString();
                record["idx"] = InputFilePath + "." + to_string(moduleIdx) + "."+ p.getUnoptFunction()->getName().str();
                records.push_back(record);
            }
            writeRecordsIntoJson(records, outfile);
        }
    }

    infile.close();
    outfile.close();
}
