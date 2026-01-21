// ChatGPT version of original python file RootFinder.py

#include "RootFinder.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

static inline double NaN() {
    return std::numeric_limits<double>::quiet_NaN();
}

GodTierRootFinder::GodTierRootFinder(
    Func func,
    double brentTol,
    int brentIter,
    double tol,
    double dx,
    double offset,
    int maxIter,
    double /*rhobeg_unused*/,
    int maxNumberOfRoots
)
    : func_(std::move(func)),
      BrentTol_(brentTol),
      BrentIter_(brentIter),
      Tol_(tol),
      dx_(dx),
      Offset_(offset),
      MaxIter_(maxIter),
      Max_Number_of_Roots_(maxNumberOfRoots) {}

bool GodTierRootFinder::IsFinite(double x) {
    return std::isfinite(x);
}

double GodTierRootFinder::Derivative(double x) const {
    const double xl = x - 0.5 * dx_;
    const double xr = x + 0.5 * dx_;
    const double fl = func_(xl);
    const double fr = func_(xr);
    return (fr - fl) / dx_;
}

void GodTierRootFinder::StopCondition(double /*a*/, double /*b*/, double turnA, double turnB) {
    // Mirrors your logic loosely:
    // stop if turnA surpasses turnB or they get too close (within 2*Offset)
    if (turnA > turnB) {
        Stop_ = true;
    } else if (std::abs(turnA - turnB) < 2.0 * Offset_) {
        Stop_ = true;
    } else {
        Stop_ = false;
    }
}

double GodTierRootFinder::GoldenSectionMin(
    const std::function<double(double)>& f,
    double lo, double hi,
    int maxIter, double tol
) const {
    // Robust, dependency-free 1D minimizer for unimodal-ish regions.
    // If f isn't unimodal, this is an approximation (which is fine per your request).
    const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
    const double resphi = 2.0 - phi; // ~0.381966

    double a = lo, b = hi;
    double c = b - resphi * (b - a);
    double d = a + resphi * (b - a);

    double fc = f(c);
    double fd = f(d);

    for (int i = 0; i < maxIter; ++i) {
        if (std::abs(b - a) < tol) break;

        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - resphi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + resphi * (b - a);
            fd = f(d);
        }
    }
    return (a + b) * 0.5;
}

double GodTierRootFinder::FindTurnPoint(double x0, double lo, double hi, bool wantMinimum) const {
    // We keep it simple:
    // - If wantMinimum: minimize f on [lo,hi]
    // - If wantMaximum: minimize (-f) on [lo,hi]
    // x0 is unused in golden-section; kept for similarity to Python design.
    (void)x0;

    if (lo > hi) std::swap(lo, hi);
    lo = std::max(lo, -std::numeric_limits<double>::max());
    hi = std::min(hi,  std::numeric_limits<double>::max());

    if (wantMinimum) {
        return GoldenSectionMin([&](double x) { return func_(x); }, lo, hi, MaxIter_, Tol_);
    }
    return GoldenSectionMin([&](double x) { return -func_(x); }, lo, hi, MaxIter_, Tol_);
}

double GodTierRootFinder::BrentRoot(double ax, double bx) const {
    // Classic Brent method for root finding (bracketed).
    // Requires f(ax)*f(bx) <= 0 and finite values.
    double a = ax, b = bx;
    double fa = func_(a);
    double fb = func_(b);

    if (!IsFinite(fa) || !IsFinite(fb)) return NaN();
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa * fb > 0.0) return NaN(); // not bracketed

    double c = a;
    double fc = fa;
    double d = b - a;
    double e = d;

    for (int iter = 0; iter < BrentIter_; ++iter) {
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }

        const double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * BrentTol_;
        const double xm = 0.5 * (c - b);

        if (std::abs(xm) <= tol1 || fb == 0.0) {
            return b;
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            // attempt inverse quadratic interpolation
            double s = fb / fa;
            double p, q;

            if (a == c) {
                // secant
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                // inverse quadratic interpolation
                double q1 = fa / fc;
                double r = fb / fc;
                p = s * (2.0 * xm * q1 * (q1 - r) - (b - a) * (r - 1.0));
                q = (q1 - 1.0) * (r - 1.0) * (s - 1.0);
            }

            if (p > 0) q = -q;
            p = std::abs(p);

            const double min1 = 3.0 * xm * q - std::abs(tol1 * q);
            const double min2 = std::abs(e * q);

            if (2.0 * p < (min1 < min2 ? min1 : min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;
        b += (std::abs(d) > tol1 ? d : (xm > 0 ? tol1 : -tol1));
        fb = func_(b);

        if (!IsFinite(fb)) return NaN();
    }

    return b; // best effort
}

std::vector<double> GodTierRootFinder::BrentIteration(
    double aOld, double bOld,
    double aNew, double bNew,
    double fAOld, double fBOld,
    double fANew, double fBNew
) const {
    std::vector<double> roots;

    // left segment: [aOld, aNew]
    if (IsFinite(fAOld) && IsFinite(fANew) && fAOld * fANew < 0.0) {
        double r = BrentRoot(aOld, aNew);
        if (IsFinite(r)) roots.push_back(r);
    }

    // right segment: [bNew, bOld]
    if (IsFinite(fBOld) && IsFinite(fBNew) && fBOld * fBNew < 0.0) {
        double r = BrentRoot(bNew, bOld);
        if (IsFinite(r)) roots.push_back(r);
    }

    return roots;
}

std::vector<double> GodTierRootFinder::UniqueSorted(const std::vector<double>& sorted, double eps) {
    std::vector<double> out;
    out.reserve(sorted.size());
    for (double x : sorted) {
        if (!IsFinite(x)) continue;
        if (out.empty() || std::abs(x - out.back()) > eps) out.push_back(x);
    }
    return out;
}

GodTierRootFinder::Step GodTierRootFinder::InitialStep(double a, double b, double f_a, double f_b) {
    Step st;
    st.a = a; st.b = b;

    const double df_a = Derivative(a);
    const double df_b = Derivative(b);

    // Match your “if derivative sign implies min vs max” idea (loosely).
    bool wantMinA = (df_a <= 0.0);
    bool wantMinB = (df_b >= 0.0);

    st.turnTypeA = wantMinA ? "Minimum" : "Maximum";
    st.turnTypeB = wantMinB ? "Minimum" : "Maximum";

    // turning points inside [a,b]
    st.aNew = FindTurnPoint(a, a, b, wantMinA);
    st.bNew = FindTurnPoint(b, a, b, wantMinB);

    st.fANew = func_(st.aNew);
    st.fBNew = func_(st.bNew);

    st.roots = BrentIteration(a, b, st.aNew, st.bNew, f_a, f_b, st.fANew, st.fBNew);

    StopCondition(a, b, st.aNew, st.bNew);
    st.finished = Stop_;
    return st;
}

GodTierRootFinder::Step GodTierRootFinder::OffsetCompression(const Step& prev) {
    Step st;
    st.a = prev.aNew;
    st.b = prev.bNew;

    // compress inward
    double aIn = st.a + Offset_;
    double bIn = st.b - Offset_;

    if (aIn > bIn) {
        st.finished = true;
        return st;
    }

    // preserve "turn type" behavior: if last time you looked for a min, now look for max (and vice versa)
    // This mimics your Python Offset_Compression flipping objective based on TurnType.
    bool wantMinA = (prev.turnTypeA == "Maximum"); // flip
    bool wantMinB = (prev.turnTypeB == "Maximum"); // flip

    st.turnTypeA = wantMinA ? "Minimum" : "Maximum";
    st.turnTypeB = wantMinB ? "Minimum" : "Maximum";

    st.aNew = FindTurnPoint(aIn, aIn, bIn, wantMinA);
    st.bNew = FindTurnPoint(bIn, aIn, bIn, wantMinB);

    st.fANew = func_(st.aNew);
    st.fBNew = func_(st.bNew);

    // Use previous endpoints for sign-change tests (rough analogue to your BrentIteration usage)
    st.roots = BrentIteration(st.a, st.b, st.aNew, st.bNew, func_(st.a), func_(st.b), st.fANew, st.fBNew);

    StopCondition(st.a, st.b, st.aNew, st.bNew);
    st.finished = Stop_;
    return st;
}

std::vector<double> GodTierRootFinder::Roots(double a, double b, double f_a, double f_b) {
    Stop_ = false;

    std::vector<double> roots;
    roots.reserve(static_cast<size_t>(Max_Number_of_Roots_));

    Step st = InitialStep(a, b, f_a, f_b);
    for (double r : st.roots) roots.push_back(r);

    // Iterate compressions
    int guard = 0;
    while (!st.finished && static_cast<int>(roots.size()) < Max_Number_of_Roots_ && guard < 10000) {
        Step next = OffsetCompression(st);

        for (double r : next.roots) {
            roots.push_back(r);
            if (static_cast<int>(roots.size()) >= Max_Number_of_Roots_) break;
        }

        st = next;
        ++guard;
    }

    // Final bracket check: if the last updated bounds still bracket a root, try Brent once more
    if (static_cast<int>(roots.size()) < Max_Number_of_Roots_) {
        const double fa = func_(st.aNew);
        const double fb = func_(st.bNew);
        if (IsFinite(fa) && IsFinite(fb) && fa * fb < 0.0) {
            double r = BrentRoot(st.aNew, st.bNew);
            if (IsFinite(r)) roots.push_back(r);
        }
    }

    // Sort + unique
    std::sort(roots.begin(), roots.end());
    roots = UniqueSorted(roots, 1e-5);

    // Pad to Max_Number_of_Roots_
    while (static_cast<int>(roots.size()) < Max_Number_of_Roots_) {
        roots.push_back(NaN());
    }
    if (static_cast<int>(roots.size()) > Max_Number_of_Roots_) {
        roots.resize(static_cast<size_t>(Max_Number_of_Roots_));
    }

    return roots;
}
