#include "utils.h"

#include <iostream>
#include <fstream>
#include <json/json.h>
#include <vector>
#include <dirent.h>
#include<regex>

using namespace llvm;

std::unique_ptr<Module> getLLVMIR(std::string IRStr) {
  std::unique_ptr<MemoryBuffer> mem = MemoryBuffer::getMemBuffer(IRStr);
  MemoryBufferRef memRef = mem->getMemBufferRef();

  SMDiagnostic err;
  static LLVMContext context;
  auto M = parseIR(memRef, err, context);

  if (!M) {
    err.print(IRStr.c_str(), outs());
    throw "WARNING: the module ir cannot be parsed!"; // 抛出运行时异常
  }
  return M;
}

std::unique_ptr<Module> getLLVMIRFromFile(std::string filepath) {
  SMDiagnostic err;
  static LLVMContext context;
  auto M = parseIRFile(filepath, err, context);

  if (!M) {
    err.print("readLLFile", outs());
    throw "WARNING: the module ir cannot be parsed!"; // 抛出运行时异常
  }
  return M;
}

void writeRecordsIntoJson(std::vector<std::map<std::string, std::string>> &records, 
                          std::ofstream &outfile){
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    Json::StreamWriter* writer = builder.newStreamWriter();
    for(auto rcd: records){
        Json::Value root;
        for(auto &KV: rcd){
            root[KV.first] = KV.second;
        }
        writer->write(root, &outfile);
        outfile<<"\n";
    }
}

std::vector<std::string> getAllFiles(const std::string& directory) {
    std::vector<std::string> filenames;
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(directory.c_str())))
        return filenames;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(directory + "/" + ptr->d_name);
    }
    closedir(pDir);
    return filenames;
}

std::vector<std::string> stringSplit(const std::string& str, char delim) {
  std::string s;
  s.append(1, delim);
  std::regex reg(s);
  std::vector<std::string> elems(std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                  std::sregex_token_iterator());
  return elems;
}

std::string getVectorAsStr(Vector vec){
  std::string res = "";
  for(auto i: vec){
    if ((i <= 0.0001 && i > 0) || (i < 0 && i >= -0.0001))
      i = 0;
    res += std::to_string(i) + ",";
  }
  return res;
}