#include <array>
#include <vector>
#include <iostream>
#include <chrono>

#include "exprtk.hpp"
#include "Simulate.hpp"
#include "utils.hpp"
#include "IncludeParams.hpp"


int main(int argc, char* argv[]) {
    
    // start global timing
    auto t0 = std::chrono::steady_clock::now();
    
    // require input.txt file
    if (argc < 2) {
        std::cerr << "Usage: echo <input_file>\n";
        return 1;
    }

    // parse input arguments
    SimConfig cfg = read_input(argv[1]);

    // construct simulator
    Simulate sim(cfg);

    // compute wake at time(s)
    std::vector<double> ComputeTimes = cfg.compute_times;
    double t_final = *std::max_element(ComputeTimes.begin(), ComputeTimes.end());

    if (!cfg.live){
        // fixed trajectory mode:

        auto T = linspace(cfg.time_i, t_final, cfg.Ntraj);
        std::vector<double> M(cfg.Ntraj, cfg.mass_i);
        std::vector<std::array<double,3>> X, V, A;
        
        // Trajecory evaluation based on input file
        typedef exprtk::symbol_table<double> sym_table_t;
        typedef exprtk::expression<double>   expr_t;
        typedef exprtk::parser<double>       parser_t;

        double t_val = cfg.time_i;
        sym_table_t sym;
        sym.add_variable("t", t_val);
        sym.add_constants();  // registers pi, e, etc.

        // compile all 9 expressions once
        auto compile = [&](const std::string& str) {
            expr_t e; e.register_symbol_table(sym);
            parser_t p;
            if (!p.compile(str, e))
                throw std::runtime_error("Bad expression: " + str);
            return e;
        };

        auto eXx = compile(cfg.Xx), eXy = compile(cfg.Xy), eXz = compile(cfg.Xz);
        auto eVx = compile(cfg.Vx), eVy = compile(cfg.Vy), eVz = compile(cfg.Vz);
        auto eAx = compile(cfg.Ax), eAy = compile(cfg.Ay), eAz = compile(cfg.Az);

        for (double t : T) {
            t_val = t;  // all expressions share this variable by reference
            X.push_back({ eXx.value(), eXy.value(), eXz.value() });
            V.push_back({ eVx.value(), eVy.value(), eVz.value() });
            A.push_back({ eAx.value(), eAy.value(), eAz.value() });
        }
        
        // run sim for fixed trajectory
        sim.run_fixed(ComputeTimes, T, M, X, V, A);
    }

    else {
        // live trajectory mode:
        // live ODE integration — not yet implemented
    }

    // end time
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "run complete at time: " << dt.count() << " s\n";
    return 0;
}
