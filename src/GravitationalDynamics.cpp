// ===============================================================
// N-body gravitational integrator
// ===============================================================






















// CHANGE THIS . JUST TAKE UPDATED POSITIONS AND COMPUTE FORCE INDIVIDUALLY ON A GIVEN ROOT 
// void ComputeGravitationalForce(
//     std::vector<Particle>& particles,
//     double G, 
//     double eps2){

//     // zero accelerations
//     for (auto& p : particles) {
//         p.a = {0.0, 0.0, 0.0};
//     }

//     const std::size_t N = particles.size();
//     for (std::size_t i = 0; i < N; ++i) {
//         for (std::size_t j = i + 1; j < N; ++j) {
//             double dx0 = particles[j].x[0] - particles[i].x[0];
//             double dx1 = particles[j].x[1] - particles[i].x[1];
//             double dx2 = particles[j].x[2] - particles[i].x[2];

//             double r2 = dx0*dx0 + dx1*dx1 + dx2*dx2 + eps2;
//             double inv_r = 1.0 / std::sqrt(r2);
//             double inv_r3 = inv_r / r2; // 1 / r^3

//             // accelerations -- so note that they are not minus each other
//             double s_i = G * particles[j].m * inv_r3; // accel on i due to j
//             double s_j = G * particles[i].m * inv_r3; // accel on j due to i

//             particles[i].a[0] += s_i * dx0;
//             particles[i].a[1] += s_i * dx1;
//             particles[i].a[2] += s_i * dx2;

//             particles[j].a[0] -= s_j * dx0;
//             particles[j].a[1] -= s_j * dx1;
//             particles[j].a[2] -= s_j * dx2;
//         }
//     }
// }

// ===============================================================
// Time integration for gase forces
// ===============================================================


// ===============================================================
// Time integration for N-body forces
// ===============================================================

void KDK_ParticleIntegrator(
    std::vector<Particle>& particles,
    double dt, 
    double G, 
    double eps2){

    // 1) Kick at half timestep
    const double half = 0.5 * dt;
    for (auto& p : particles) {
        p.v[0] += half * p.a[0];
        p.v[1] += half * p.a[1];
        p.v[2] += half * p.a[2];
    }

    // 2) Drift at full timestep
    for (auto& p : particles) {
        p.x[0] += dt * p.v[0];
        p.x[1] += dt * p.v[1];
        p.x[2] += dt * p.v[2];
    }

    // 3) Recompute acceleration at new positions: a(t+dt)
    ComputeGravitationalForce(particles, G, eps2); // Careful. Do we want for a given particle or all of them?

    // 4) Kick: v(t+dt) = v(t+dt/2) + (dt/2)*a(t+dt)
    for (auto& p : particles) {
        p.v[0] += half * p.a[0];
        p.v[1] += half * p.a[1];
        p.v[2] += half * p.a[2];
    }

    // IMPORTANT add_event to particle classes ie. p.add_event()
};