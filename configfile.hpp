#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "configfile.h"

/* Constructor of the configfile class */
inline configfile::configfile(const std::string& fileName) {
    std::ifstream file(fileName.c_str());
    if (!file) {
        std::cerr << "[configfile] Impossible d'ouvrir le fichier " << fileName << std::endl;
    } else {
        std::string lineread;
        while (getline(file, lineread)) {
            process(lineread);
        }
        file.close();
    }
}

// Destructor
inline configfile::~configfile() {}

/* Method for printing data in string format in an output file */
inline void configfile::printOut(const std::string& path) const {
    std::ofstream outputFile(path.c_str());
    if (outputFile.is_open()) {
        outputFile << toString() << std::endl;
    }
    outputFile.close();
}

/* Generate a string of the form: string1 = string2 \n */
inline std::string configfile::toString() const {
    std::string strToReturn;
    for (std::map<std::string,std::string>::const_iterator iter = configMap.begin();
         iter != configMap.end(); ++iter) {
        strToReturn.append(iter->first);
        strToReturn.append("=");
        strToReturn.append(iter->second);
        strToReturn.append("\n");
    }
    return strToReturn;
}

/* Extract information from a line and update configMap */
inline void configfile::process(const std::string& lineread) {
    size_t commentPosition = trim(lineread).find('%',0);
    if (commentPosition != 0 && trim(lineread).length() > 0) {
        size_t equalPosition = lineread.find('=',1);
        if (equalPosition == std::string::npos) {
            std::cerr << "[configfile] Ligne sans '=' : \"" << trim(lineread) << "\"" << std::endl;
        } else {
            std::string key = trim(lineread.substr(0,equalPosition));
            std::string value = trim(lineread.substr(equalPosition+1,lineread.length()));
            std::map<std::string, std::string>::const_iterator val = configMap.find(key);
            if (val != configMap.end()) {
                configMap.erase(key);
            }
            configMap.insert(std::pair<std::string,std::string>(key,value));
        }
    }
}

/* Remove tabs and spaces at the beginning and end of a string */
inline std::string configfile::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\r");
    return str.substr(first, (last-first+1));
}   

/* Generic template get<T> */
template<typename T>
inline T configfile::get(const std::string& key, const T& initValue) const {
    std::map<std::string, std::string>::const_iterator val = configMap.find(key);
    T out(initValue);
    if (val != configMap.end()) {
        std::istringstream iss(val->second);
        iss >> out;
    } else {
        std::cerr << "[configfile] Paramètre manquant : " << key << std::endl;
    }
    return out;
}

/* Specialization for bool */
template<>
inline bool configfile::get<bool>(const std::string& key, const bool& initValue) const {
    std::istringstream iss(configMap.find(key)->second);
    bool result(initValue);
    iss >> result;
    if (iss.fail()) {
        iss.clear();
        iss >> std::boolalpha >> result;
    }
    return result;
}
