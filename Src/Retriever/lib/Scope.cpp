#include "Scope.h"

#include <map>

#include "llvm/IR/InstIterator.h"

using namespace llvm;
using namespace std;
Scope* constructScope(Function *F){
    // init
    map<DIScope*, Scope*> di2Scope;
    di2Scope.clear();
    // F->print(errs());
    for (auto &I : instructions(F)) {
        if(I.getDebugLoc()){
            // I.print(errs()); errs()<<"\n";
            MDNode *node = I.getDebugLoc().getScope();
            if(DIScope *diScope = dyn_cast<DILocalScope>(node)){
                Scope* scope;
                if(di2Scope.count(diScope)){
                    scope = di2Scope[diScope];
                }else{
                    di2Scope[diScope] = new Scope(diScope);
                    scope = di2Scope[diScope];
                }
                scope->addBB(I.getParent());
            }
        }
    }

    // for(auto &KV: di2Scope){
    //     KV.first->print(errs());
    //     errs()<<"\n";
    // }

    // construct the relationship(child/parent)
    bool changed = true;
    while(changed){
        changed = false;
        for(auto &KV: di2Scope){
            DIScope* discope = KV.first;
            Scope* scope = KV.second;
            DIScope* parentDISScope = discope->getScope();
            if(parentDISScope && isa<DILocalScope>(parentDISScope)){
                Scope* parentScope;
                if(di2Scope.count(parentDISScope)){
                    parentScope = di2Scope[parentDISScope];
                }else{
                    parentScope = new Scope(parentDISScope);
                    di2Scope[parentDISScope] = parentScope;
                }

                if(scope->getParent()){
                    assert(scope->getParent() == parentScope && "ERROR: One Scope must have only one parent!\n");
                }
                else{
                    scope->setParent(parentScope);
                    parentScope->addChild(scope);
                    changed = true;
                    break;
                }
            }
        }
    }

    // for(auto &KV: di2Scope){
    //     KV.second->getDIScope()->print(errs());
        
    //     errs()<<"\tp:";
    //     if (KV.second->getParent())
    //         KV.second->getParent()->getDIScope()->print(errs());
    //     else
    //         errs()<<"noparent";
    //     errs()<<"\n";
    // }

    // for(auto &KV: di2Scope){
    //     KV.second->getDIScope()->print(errs());
        
    //     errs()<<"\tp:";
    //     if (KV.second->getOutmostParentScope())
    //         KV.second->getOutmostParentScope()->getDIScope()->print(errs());
    //     else
    //         errs()<<"noparent";
    //     errs()<<"\n";
    // }

    // @TODO: donnot consider about flto!
    Scope* root = NULL;
    for(auto &KV: di2Scope){
        Scope* s = KV.second;
        if(root && s->getOutmostParentScope()){
            assert(root == s->getOutmostParentScope() && "ERROR: we donnot consider about flto!\n");
        }
        else if(root && !s->getOutmostParentScope()){
            // errs()<<root->toString()<<"\n";
            assert(root == s && "ERROR: we donnot consider about flto!\n");
        }
        else if(!root && !s->getOutmostParentScope()){
            root = s;
        }
        else{
            root = s->getOutmostParentScope();
        }
    }
    // errs()<<"!!!!";
    // root->getDIScope()->print(errs());
    // errs()<<"\n";
    // errs()<<"!!!!"<<root->toString()<<"\n";
    return root;
}