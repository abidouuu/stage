#include "dynamo.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <vector>
#include <random>
#include <filesystem>
namespace fs=std::filesystem;
using namespace std;

//surcharge d'opérateurs
vector<double> operator*(double a, const vector<double>& v) {
    vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = a * v[i];
    }
    return result;
}

vector<double> operator+(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("size of vectors doesn't match");
    }

    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

//constructeur
dynamo::dynamo(int argc, char* argv[]) {
    fs::path inputpath("configuration.in");
    if (argc > 1) {
        inputpath = fs::path(argv[1]) / "configuration.in"; 
    }

    configfile configfile(inputpath.string()); 

    for (int i(2); i < argc; i++) {
        configfile.process(argv[i]);
    }

    B0 = configfile.get<double>("B0");
    b0 = configfile.get<double>("b0");

    epsiloneq = configfile.get<double>("epsiloneq");
    Lambda = configfile.get<double>("Lambda");
    kappaeq = configfile.get<double>("kappaeq");

    tauepsilon = configfile.get<double>("tauepsilon");
    taukappa = configfile.get<double>("taukappa");
    deltaepsilon = configfile.get<double>("deltaepsilon");
    deltakappa = configfile.get<double>("deltakappa");

    inter_kappa = configfile.get<bool>("inter_kappa");
    inter_epsilon = configfile.get<bool>("inter_epsilon");

    term = configfile.get<string>("term");

    dt = configfile.get<double>("dt");
    tfin = configfile.get<double>("tfin");

    if (argc > 1) {
        fs::path outputpath = fs::path(argv[1]) / "output.txt";
        outputfile = make_unique<ofstream>(outputpath);
    }if (!outputfile->is_open()) {
        cerr << "Erreur : impossible d'ouvrir output.txt" << endl;
        exit(1);
    }
    outputfile->precision(15);
}

//destructeur
dynamo::~dynamo() {
    if (outputfile->is_open()) {
        outputfile->close();
    }
}

//ecriture dans le fichier de sortie 
void dynamo::printout(bool force){
    if (force){
        if (outputfile->is_open()) {
            if (term=="short"){
                *outputfile << t << " " << B << " " << b << endl;
            } else if (term=="mid"){
                *outputfile << t << " " << B << " " << b << " " << kappa 
                << " " << epsilon << endl;
            } else{*outputfile << t << " " << B << " " << b << " " << kappa 
                << " " << epsilon << " " << Omega << endl;

            }
        }
    }
}

//eq diff 
vector<double> dynamo::rhs(vector<double> v){
    vector<double> dot(2);

    dot[0] = (epsilon + kappa*v[1]*v[1])*v[0] - Lambda*v[0]*v[0]*v[0] ; //Bdot
    dot[1] = v[1] - v[1]*v[1]*v[1] - kappa*v[1]*v[1]*v[0] ; //bdot

    return dot;
}

//évolution de b et B
void dynamo::rk4(){
    vector<double> k1(2), k2(2), k3(2), k4(2), v(2);
    v[0]=B;
    v[1]=b;
    
    k1 = rhs(v);
    k2 = rhs(v + (dt*0.5)*k1);
    k3 = rhs(v + (dt*0.5)*k2);
    k4 = rhs(v + dt*k3);

    B += (dt/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
    b += (dt/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
}

//évolution des paramètres stochastiques (ornstein-uhlenbeck)
void dynamo::stochastic(double &X, double Xeq, double tauX, double deltaX){
    static normal_distribution<> N01(0.0, sqrt(dt)); // (moyene , déviation standard)
    static mt19937 gen(std::random_device{}());
    double dW = N01(gen);
    X += -(1/tauX) * (X - Xeq) * dt + sqrt(deltaX) * dW;
}

void dynamo::skumanich()
{
    double t_ini_Gyr = 1.0;
    double t_fin_Gyr = 10.0;
    double t_sun_Gyr = 4.6;

    double n_rot = 0.5;

    double eps_ref = epsiloneq;

    double eps_ini = 1.0;

    double age_Gyr = t_ini_Gyr
                   + (t_fin_Gyr - t_ini_Gyr) * (t / tfin);

    Omega = pow(age_Gyr / t_sun_Gyr, -0.5);

    double p = (n_rot + 1.0) / 2.0;

    double fac_ini = pow(t_ini_Gyr / t_sun_Gyr, -p);

    double A = (eps_ini - eps_ref) / (fac_ini - 1.0);
    double B = eps_ref - A;

    epsilon = A * pow(age_Gyr / t_sun_Gyr, -p) + B;
}
//court terme
void dynamo::short_step(){
    rk4();
}

//moyen terme
void dynamo::mid_step(){
    short_step();
    if (inter_epsilon and term!="long") {
        stochastic(epsilon, epsiloneq, tauepsilon, deltaepsilon);
    }
    if (inter_kappa){
        stochastic(kappa, kappaeq, taukappa, deltakappa);
    }
}

//long terme
void dynamo::long_step(){
    mid_step();
    skumanich();
}

//évolution tout terme
void dynamo::run_step(void (dynamo::*step_funct)()){

    while(t<tfin){
        if (B<1e-15) B=1e-15;
        (this->*step_funct)();
        t += dt;
        step ++;
        printout(true);
    }
}

//simulation
void dynamo::run(){
    t=0;
    step=0;

    B=B0;
    b=b0;

    epsilon=epsiloneq;
    kappa=kappaeq;

    Omega=2.14476105895272;

    if (term=="short") {run_step(&dynamo::short_step);}
    if (term=="mid") {run_step(&dynamo::mid_step);}
    if (term=="long") {
        epsilon=1;
        inter_epsilon=false;
        run_step(&dynamo::long_step);
    }

    printout(true);
}
