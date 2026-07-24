#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <functional>
#include "configfile.h"
#include <valarray>
#include <algorithm>
#include <string>
#include <cmath>
#include <memory> // pour std::unique_ptr

using namespace std;

class dynamo{
    private :
        double B,b ; //champs magnétiques

        double epsilon, Lambda, kappa ; //paramètres fondamentaux
        double tauepsilon, taukappa, deltaepsilon, deltakappa ; //paramètres d'intermittence
        bool inter_kappa, inter_epsilon ; // faire varier epsilon/kappa ou non
        double Omega ; //vitesse angulaire

        string term ; //"long", "mid", "short"

        double B0, b0, sigmaB0, epsiloneq, kappaeq ; //conditions initiales
        double dt, t, tfin ; //paramètres de temps
        int step ; //nombre de pas de temps effectués

        vector<double> rhs(vector<double> v); //éq diff
        void rk4(); //évolution de b et B
        void stochastic(double &X, double X0, double thetaX, double deltaX); //évolution des paramètres stochastiques
        void skumanich(); //évolution de sigmaB (vents solaires)

        void short_step(); //évolution court terme
        void mid_step(); //évolution moyen terme
        void long_step(); //évolution long terme
        void run_step(void (dynamo::*step_funct)());

        void printout(bool force=false); //écriture dans le fichier de sortie
        unique_ptr<ofstream> outputfile; //fichier de sortie

    public : 
        dynamo(int argc, char* argv[]); //constructeur
        ~dynamo(); //destructeur
        void run(); //simu
};