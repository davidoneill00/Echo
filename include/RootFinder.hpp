#pragma once

#include <functional>
#include <vector>
#include <string>

class GodTierRootFinder {
public:
    using Func = std::function<double(double)>;

    GodTierRootFinder(
        Func func,
        double brentTol,
        int brentIter,
        double tol,
        double dx,
        double offset,
        int maxIter,
        double /*rhobeg_unused*/,
        int maxNumberOfRoots
    );

    // Returns up to Max_Number_of_Roots roots (sorted), padded with NaN if fewer.
    // f_a and f_b are optional hints; if you don't have them, just pass func(a), func(b).
    std::vector<double> Roots(double a, double b, double f_a, double f_b);

private:
    struct Step {
        double a = 0.0, b = 0.0;
        double aNew = 0.0, bNew = 0.0;
        double fANew = 0.0, fBNew = 0.0;
        std::string turnTypeA;
        std::string turnTypeB;
        bool finished = false;
        std::vector<double> roots;
    };

    Func func_;
    double BrentTol_ = 1e-10;
    int BrentIter_ = 100;
    double Tol_ = 1e-8;     // for minimization and internal comparisons
    double dx_ = 1e-6;      // derivative finite difference
    double Offset_ = 1e-3;  // compression offset
    int MaxIter_ = 200;     // minimization iterations
    int Max_Number_of_Roots_ = 10;

    bool Stop_ = false;

private:
    // Core helpers
    double Derivative(double x) const;
    void StopCondition(double a, double b, double turnA, double turnB);

    // Brent root finder on [a,b] assuming sign change.
    double BrentRoot(double a, double b) const;

    // Golden-section minimize on [lo, hi]
    double GoldenSectionMin(const std::function<double(double)>& f,
                            double lo, double hi,
                            int maxIter, double tol) const;

    // Find a “turning point” near a side by minimizing or maximizing on [lo,hi].
    // We keep it simple: minimize f or minimize -f.
    double FindTurnPoint(double x0, double lo, double hi, bool wantMinimum) const;

    // If f changes sign between old and new endpoints, run Brent to get root(s).
    std::vector<double> BrentIteration(
        double aOld, double bOld,
        double aNew, double bNew,
        double fAOld, double fBOld,
        double fANew, double fBNew
    ) const;

    // De-duplicate sorted roots
    static std::vector<double> UniqueSorted(const std::vector<double>& sorted, double eps);

    Step InitialStep(double a, double b, double f_a, double f_b);
    Step OffsetCompression(const Step& prev);

    // Utility
    static bool IsFinite(double x);
};
