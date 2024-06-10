#ifndef __SCOPE__
#define __SCOPE__

#include <set>
#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/DebugInfoMetadata.h"
using namespace std;

class Scope{
    set<llvm::BasicBlock*> BBs;
    llvm::DIScope *scope;
    Scope *Parent;
    set<Scope*> Childs;
public:
    Scope(llvm::DIScope *scope): BBs({}), scope(scope), Parent(NULL), Childs({}){}
    ~Scope(){
        // llvm::errs()<<"DELETE: "<<toString()<<"\n";
        for(Scope* c: Childs){
            delete c;
        }
    }
    llvm::DIScope* getDIScope(){ return scope; }
    Scope* getParent(){ return Parent; }
    set<llvm::BasicBlock*> getBBs(){
        set<llvm::BasicBlock*> allBBs;
        allBBs.insert(BBs.begin(), BBs.end());
        for(Scope* c: Childs){
            set<llvm::BasicBlock*> tBBs = c->getBBs();
            allBBs.insert(tBBs.begin(), tBBs.end());
        }
        return allBBs;
    }

    unsigned getLine(){
        if(llvm::DILexicalBlock *lb = llvm::dyn_cast<llvm::DILexicalBlock>(scope))
            return lb->getLine();

        if(!llvm::isa<llvm::DISubprogram>(scope) && 
            !llvm::isa<llvm::DILexicalBlockFile>(scope))
            llvm::errs()<<"WARNING: "<<toString()<<"\n";
        return 0;
    }

    unsigned getColumn(){
        if(llvm::DILexicalBlock *lb = llvm::dyn_cast<llvm::DILexicalBlock>(scope))
            return lb->getColumn();

        if(!llvm::isa<llvm::DISubprogram>(scope) && 
            !llvm::isa<llvm::DILexicalBlockFile>(scope)){
            // (*BBs.begin())->getParent()->print(llvm::errs());
            // llvm::errs()<<"WARNING: ";
            // scope->print(llvm::errs());
            // llvm::errs()<<"\n";
            // assert(0);
            llvm::errs()<<"WARNING: "<<toString()<<"\n";
        }
        return 0;
    }

    std::string getFileName(){
        return scope->getFilename().str(); 
    }

    void addBB(llvm::BasicBlock* BB){ BBs.insert(BB); }
    void addBBs(set<llvm::BasicBlock*> newBBs){
        for(llvm::BasicBlock* BB: newBBs){
            BBs.insert(BB);
        }
    }

    void setParent(Scope *s){ Parent = s; }
    void addChild(Scope *s){
        if(Childs.count(s))
            return;
        Childs.insert(s);
        // addBBs(s->getBBs());
    }

    set<Scope*> getInnermostScopes(){
        set<Scope*> res;

        if(Childs.size() == 0){
            res.insert(this);
            return res;
        }

        for(Scope* s: Childs){
            set<Scope*> innermostSs = s->getInnermostScopes();
            for(Scope* is: innermostSs)
                res.insert(is);
        }
        return res;
    }

    Scope* getFirstDILocalParentScope(){
        Scope *curr = Parent;
        while(curr){
            if(llvm::isa<llvm::DILocalScope>(curr->getDIScope()))
                return curr;
            curr = curr->getParent();
        }
        return NULL;
    }

    Scope* getOutmostParentScope(){
        Scope *curr = Parent;
        if(!curr)
            return NULL;
        while(1){
            if(!curr->getParent()){
                return curr;
            }
            curr = curr->getParent();
        }
        return NULL;
    }

    std::string toString(){
        // scope->print(llvm::errs());
        // llvm::errs()<<" SubScope:{";
        // for(auto c: Childs){
        //     c->toString();
        // }
        // llvm::errs()<<"}";

        std::vector<std::string> BBNames;
        // if(llvm::isa<llvm::DISubprogram>(scope) || 
        //     llvm::isa<llvm::DILexicalBlockFile>(scope)){ // to prevent there is a bb only contains a br without dbgInfo
        //     llvm::Function *F = (*getBBs().begin())->getParent();
        //     for(auto &BB: *F){
        //         std::string Name = "";
        //         llvm::raw_string_ostream OS(Name);
        //         BB.printAsOperand(OS);
        //         BBNames.push_back(Name);
        //     }
        // }else{
        for(auto BB: getBBs()){
            std::string Name = "";
            llvm::raw_string_ostream OS(Name);
            BB->printAsOperand(OS);
            BBNames.push_back(Name);
        }
        // }
        
        std::sort(BBNames.begin(), BBNames.end());

        std::string ans = "[";
        for(auto BB: BBNames){
            ans += BB + ",";
        }
        ans += "]";
        return ans;
    }
};

Scope* constructScope(llvm::Function *F);

#endif