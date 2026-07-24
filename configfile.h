// Classe facilitant la lecture de fichiers de configuration.
// Contributeurs : K. Steiner, J. Dominski, N. Ohana, C. Sommariva
// Utilisation : Envoyer au constructeur le nom d'un fichier contenant
// les parametres sous la forme [param=valeur] sur chaque ligne, puis
// appeler get<type>("param") pour acceder a un parametre.

#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include <map>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

/* This class is useful for reading input files in ASCII and
 writing output files in ASCII */
class configfile{

  public:
      /* Constructor of the configfile class
         inputs:
           filename: (str) string containing the file name to read
      */
      configfile(const std::string& filename);

      /* destructor of the configfile class*/
      ~configfile();

      /* Retrun a boolean from key or the current input string state
         if casting fails
         inputs:
           key:      (str) map key of the desired element
           initValue (T)(optional) initial value in case the reading 
			     procdedure fails, default: 0/NULL
         outputs:
           result: (bool) boolean value from input is available of 
                          input string state otherwise
         Retrun a boolean from from key or the current input string state
         if casting fails
         inputs:
           key:      (str) map key of the desired element
           initValue (T)(optional) initial value in case the reading 
      			           procdedure fails, default: 0/NULL
         outputs:
           result: (bool) boolean value from input is available of 
                          input string state otherwise
      */      
      template<typename T> T get(const std::string& key,\
      const T& initValue=T()) const;

      /* The method extract information for a read string add the LHS 
         as key value in configMap map and RHS as value
         inputs:
           lineread: (str) string line read from file
      */
      void process(const std::string& lineread);

      /* Generate a string of the form: string1 = string2 \n */
      std::string toString() const;

      /* Method for printing data in string formato in an output file
         inputs:
           path: (str) string containing the path to the outputfile
      */
      void printOut(const std::string& path) const;

  private:

      /* Remove tabs and spaces at the beginning and end of a string
         input:
           str: (str) input string
         output:
           str: (sre) string without beginning / ending tabs and spaces
      */
      std::string trim(const std::string& str);

      /* Map containing the input data as string with form (key,value) */
      std::map<std::string, std::string> configMap;
};

// include template functions
#include "configfile.hpp"

#endif
