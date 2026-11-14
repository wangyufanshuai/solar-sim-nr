// B3 step 1: native port of the periodic FO-CCZ4 volume RHS.
//
// Same equations, same term grouping and the same centred finite differences as
// science/numerical_relativity/fo_ccz4_rhs.py::_fo_ccz4_volume_rhs_3d with
// gauge_profile=None (constant f and mu) on a periodic grid:
//   FD2: (f[i+1] - f[i-1]) / (2h)
//   FD4: (8 (f[i+1] - f[i-1]) - (f[i+2] - f[i-2])) / (12h)
// Non-periodic axes use the SBP(4,2) operator of science/numerical_relativity/sbp_boundary_operator.py
// (Mattsson-Nordstrom 2004: four order-2 boundary rows per face, order-4 interior), as used by
// fo_ccz4_sbp_volume.SBPVolumeOperator. The optional field-dependent gauge profile
// (fo_ccz4_gauge_profile: f = shift_scale alpha chi, mu = ordering_scale chi, plus the product-rule
// source in the B equation) matches gauge_profile != None.
// Known arithmetic differences from the Python reference include: the 3x3 inverse uses the
// adjugate formula instead of LU, and sums are accumulated in a fixed loop order.
// Built with -ffp-contract=off so no fused multiply-adds change the rounding further.
//
// Data layout: q and out are [58][nx][ny][nz] (slot-major, C order), the canonical
// FO-CCZ4 slot order: 0-5 h (ij = 00,01,02,11,12,22), 6 ln alpha, 7-9 beta, 10 ln phi
// (= ln chi / 2), 11-16 Atilde, 17 K, 18 Theta, 19-21 GammaHat, 22-24 b, 25-27 A_k,
// 28-36 B_k^i (k major), 37-54 D_kij (k major, ij symmetric as above), 55-57 P_k.

#include <cmath>
#include <cstdint>

namespace {

constexpr int SYM[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
inline int sym_index(int i, int j) {
  static const int table[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  return table[i][j];
}

struct Params {
  double lapse_coefficient, shift_coefficient, shift_damping;
  double kappa1, kappa2, kappa3, cleaning_speed, ordering_mu, shift_strength, tau_inverse, k0, c;
  double profile, shift_scale, ordering_scale;  // profile = 1: field-dependent f and mu
};

// SBP(4,2) boundary rows (left face); the right face mirrors them with opposite sign.
constexpr double SBP_ROWS[4][6] = {
    {-24.0 / 17, 59.0 / 34, -4.0 / 17, -3.0 / 34, 0, 0},
    {-1.0 / 2, 0.0, 1.0 / 2, 0, 0, 0},
    {4.0 / 43, -59.0 / 86, 0.0, 59.0 / 86, -4.0 / 43, 0},
    {3.0 / 98, 0.0, -59.0 / 98, 0.0, 32.0 / 49, -4.0 / 49}};
constexpr int SBP_LEN[4] = {4, 3, 5, 6};

struct Grid {
  int nx, ny, nz, order;
  int sbp[3];     // 1: SBP(4,2) on this axis, 0: periodic
  double h[3];
  double inv[3];  // 1/(2h) for FD2, 1/(12h) for FD4
  std::int64_t stride[3];
  int n[3];
};

// Periodic centred first derivative of slot `slot` along `axis` at (i,j,k).
inline double d1(const double* q, const Grid& g, int slot, int axis, int i, int j, int k) {
  const std::int64_t points = static_cast<std::int64_t>(g.nx) * g.ny * g.nz;
  const double* f = q + slot * points;
  int idx[3] = {i, j, k};
  auto at = [&](int offset) {
    int p[3] = {idx[0], idx[1], idx[2]};
    int m = g.n[axis];
    p[axis] = ((p[axis] + offset) % m + m) % m;
    return f[(static_cast<std::int64_t>(p[0]) * g.ny + p[1]) * g.nz + p[2]];
  };
  if (g.sbp[axis]) {
    const int m = g.n[axis], x = idx[axis];
    auto node = [&](int position) { return at(position - x); };
    if (x >= 4 && x < m - 4)  // same pairing as SBP42.derivative: ((q-2 - q+2) + 8 (q+1 - q-1)) / 12 / h
      return (node(x - 2) - node(x + 2) + 8.0 * (node(x + 1) - node(x - 1))) / 12.0 / g.h[axis];
    double v = 0.0;
    if (x < 4) {
      for (int t = 0; t < SBP_LEN[x]; ++t) v += SBP_ROWS[x][t] * (node(t) - node(x));
    } else {
      const int r = m - 1 - x;
      for (int t = 0; t < SBP_LEN[r]; ++t) v -= SBP_ROWS[r][t] * (node(m - 1 - t) - node(x));
    }
    return v / g.h[axis];
  }
  if (g.order == 4) return (8.0 * (at(1) - at(-1)) - (at(2) - at(-2))) * g.inv[axis];
  return (at(1) - at(-1)) * g.inv[axis];
}

void point_rhs(const double* q, double* out, const Grid& g, const Params& c, int i, int j, int kk, double* theta_terms = nullptr) {
  const std::int64_t points = static_cast<std::int64_t>(g.nx) * g.ny * g.nz;
  const std::int64_t at = (static_cast<std::int64_t>(i) * g.ny + j) * g.nz + kk;
  auto Q = [&](int slot) { return q[slot * points + at]; };
  auto dq = [&](int slot, int axis) { return d1(q, g, slot, axis, i, j, kk); };

  // ---- fields
  double h[3][3], At[3][3], beta[3], Gh[3], bb[3], A[3], B[3][3], D[3][3][3], P[3];
  for (int s = 0; s < 6; ++s) {
    const int a = SYM[s][0], b = SYM[s][1];
    h[a][b] = h[b][a] = Q(s);
    At[a][b] = At[b][a] = Q(11 + s);
    for (int ax = 0; ax < 3; ++ax) D[ax][a][b] = D[ax][b][a] = Q(37 + 6 * ax + s);
  }
  const double alpha = std::exp(Q(6));
  const double chi = std::exp(2.0 * Q(10));
  const double K = Q(17), Th = Q(18);
  for (int a = 0; a < 3; ++a) {
    beta[a] = Q(7 + a); Gh[a] = Q(19 + a); bb[a] = Q(22 + a); A[a] = Q(25 + a); P[a] = Q(55 + a);
    for (int b = 0; b < 3; ++b) B[a][b] = Q(28 + 3 * a + b);
  }

  // ---- derivatives (index order: [axis][field components])
  double dAt[3][3][3], dK[3], dTh[3], dGh[3][3], dbb[3][3], dA[3][3], dB[3][3][3], dD[3][3][3][3], dP[3][3];
  for (int ax = 0; ax < 3; ++ax) {
    for (int s = 0; s < 6; ++s) {
      const int a = SYM[s][0], b = SYM[s][1];
      dAt[ax][a][b] = dAt[ax][b][a] = dq(11 + s, ax);
      for (int m = 0; m < 3; ++m) dD[ax][m][a][b] = dD[ax][m][b][a] = dq(37 + 6 * m + s, ax);
    }
    dK[ax] = dq(17, ax);
    dTh[ax] = dq(18, ax);
    for (int a = 0; a < 3; ++a) {
      dGh[ax][a] = dq(19 + a, ax);
      dbb[ax][a] = dq(22 + a, ax);
      dA[ax][a] = dq(25 + a, ax);
      dP[ax][a] = dq(55 + a, ax);
      for (int b = 0; b < 3; ++b) dB[ax][a][b] = dq(28 + 3 * a + b, ax);
    }
  }

  // ---- geometry (_geometry)
  const double det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
                   - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
                   + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
  double hi[3][3];
  hi[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) / det;
  hi[0][1] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) / det;
  hi[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) / det;
  hi[1][0] = (h[1][2] * h[2][0] - h[1][0] * h[2][2]) / det;
  hi[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) / det;
  hi[1][2] = (h[0][2] * h[1][0] - h[0][0] * h[1][2]) / det;
  hi[2][0] = (h[1][0] * h[2][1] - h[1][1] * h[2][0]) / det;
  hi[2][1] = (h[0][1] * h[2][0] - h[0][0] * h[2][1]) / det;
  hi[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) / det;

  double dup[3][3][3];  // d_upup[k][i][j] = hinv[i][a] hinv[j][b] D[k][a][b]
  for (int k = 0; k < 3; ++k)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        double v = 0.0;
        for (int x = 0; x < 3; ++x)
          for (int y = 0; y < 3; ++y) v += hi[a][x] * hi[b][y] * D[k][x][y];
        dup[k][a][b] = v;
      }

  // Christoffel combination D_fsl + D_sfl - D_lfs
  auto comb = [&](int f, int s, int l) { return D[f][s][l] + D[s][f][l] - D[l][f][s]; };
  double cc[3][3][3], pc[3][3][3];
  for (int u = 0; u < 3; ++u)
    for (int f = 0; f < 3; ++f)
      for (int s = 0; s < 3; ++s) {
        double v = 0.0, w = 0.0;
        for (int l = 0; l < 3; ++l) {
          v += hi[u][l] * comb(f, s, l);
          w += hi[u][l] * (h[s][l] * P[f] + h[f][l] * P[s] - h[f][s] * P[l]);
        }
        cc[u][f][s] = v;
        pc[u][f][s] = v - w;
      }

  double dsd[3][3][3][3];  // d_(a D_b)ij
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y) dsd[a][b][x][y] = 0.5 * (dD[a][b][x][y] + dD[b][a][x][y]);
  double dsp[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) dsp[a][b] = 0.5 * (dP[a][b] + dP[b][a]);

  double dcc[3][3][3][3], dpc[3][3][3][3];
  for (int d = 0; d < 3; ++d)
    for (int u = 0; u < 3; ++u)
      for (int f = 0; f < 3; ++f)
        for (int s = 0; s < 3; ++s) {
          double algebraic = 0.0, differential = 0.0;
          for (int l = 0; l < 3; ++l) {
            algebraic += dup[d][u][l] * comb(f, s, l);
            differential += hi[u][l] * (dsd[d][f][l][s] + dsd[d][s][l][f] - dsd[d][l][f][s]);
          }
          dcc[d][u][f][s] = -2.0 * algebraic + differential;

          double first = 0.0;
          for (int m = 0; m < 3; ++m) first -= 2.0 * dup[d][u][m] * comb(f, s, m);
          double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
          for (int l = 0; l < 3; ++l) {
            t1 += dup[d][u][l] * (h[s][l] * P[f] + h[f][l] * P[s] - h[f][s] * P[l]);
            t2 += hi[u][l] * (D[d][s][l] * P[f] + D[d][f][l] * P[s] - D[d][f][s] * P[l]);
            t3 += hi[u][l] * (dsd[d][f][l][s] + dsd[d][s][l][f] - dsd[d][l][f][s]);
            t4 += hi[u][l] * (h[s][l] * dsp[d][f] + h[f][l] * dsp[d][s] - h[f][s] * dsp[d][l]);
          }
          double term = 2.0 * t1;
          term -= t2 * 2.0;
          term += t3;
          term -= t4;
          dpc[d][u][f][s] = first + term;
        }

  double ricci[3][3];
  for (int f = 0; f < 3; ++f)
    for (int s = 0; s < 3; ++s) {
      double v = 0.0;
      for (int u = 0; u < 3; ++u) {
        v += dpc[u][u][f][s] - dpc[s][u][f][u];
        for (int l = 0; l < 3; ++l) v += pc[l][f][s] * pc[u][l][u] - pc[l][f][u] * pc[u][l][s];
      }
      ricci[f][s] = v;
    }

  double tg[3], dtg[3][3];
  for (int a = 0; a < 3; ++a) {
    double v = 0.0;
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 3; ++y) v += hi[x][y] * cc[a][x][y];
    tg[a] = v;
  }
  for (int d = 0; d < 3; ++d)
    for (int u = 0; u < 3; ++u) {
      double v = 0.0;
      for (int f = 0; f < 3; ++f)
        for (int s = 0; s < 3; ++s) v += -2.0 * dup[d][f][s] * cc[u][f][s] + hi[f][s] * dcc[d][u][f][s];
      dtg[d][u] = v;
    }
  double gdiff[3], zl[3], zu[3];
  for (int a = 0; a < 3; ++a) gdiff[a] = Gh[a] - tg[a];
  for (int a = 0; a < 3; ++a) {
    double v = 0.0;
    for (int b = 0; b < 3; ++b) v += h[a][b] * gdiff[b];
    zl[a] = 0.5 * v;
    zu[a] = 0.5 * chi * gdiff[a];
  }
  double covdz[3][3];
  for (int f = 0; f < 3; ++f)
    for (int s = 0; s < 3; ++s) {
      double v = 0.0, w = 0.0, x = 0.0;
      for (int l = 0; l < 3; ++l) {
        v += D[f][s][l] * gdiff[l];
        w += h[s][l] * (dGh[f][l] - dtg[f][l]);
        x += pc[l][f][s] * zl[l];
      }
      covdz[f][s] = v + 0.5 * w - x;
    }
  double rz[3][3];
  double scalar_sum = 0.0;
  for (int f = 0; f < 3; ++f)
    for (int s = 0; s < 3; ++s) rz[f][s] = ricci[f][s] + covdz[f][s] + covdz[s][f];
  for (int f = 0; f < 3; ++f)
    for (int s = 0; s < 3; ++s) scalar_sum += hi[f][s] * rz[f][s];
  const double scalar = chi * scalar_sum;

  // ---- volume RHS (_fo_ccz4_volume_rhs_3d)
  const bool profile = c.profile != 0.0;
  const double shift_f = profile ? c.shift_scale * alpha * chi : c.shift_coefficient;
  const double ordering_mu = profile ? c.ordering_scale * chi : c.ordering_mu;
  const double sS = c.shift_strength, G = c.lapse_coefficient, cc_ = c.c, tau = c.tau_inverse;
  double trA = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) trA += hi[a][b] * At[a][b];
  double aup[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      double v = 0.0;
      for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y) v += hi[a][x] * hi[b][y] * At[x][y];
      aup[a][b] = v;
    }
  const double divb = B[0][0] + B[1][1] + B[2][2];
  double hess[3][3];
  for (int f = 0; f < 3; ++f)
    for (int s = 0; s < 3; ++s) {
      double v = 0.0;
      for (int k = 0; k < 3; ++k) v += pc[k][f][s] * A[k];
      hess[f][s] = alpha * (A[f] * A[s] - v + 0.5 * (dA[f][s] + dA[s][f]));
    }
  double lap_sum = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) lap_sum += hi[a][b] * hess[a][b];
  const double lap_hess = chi * lap_sum;
  double dbs[3][3][3];  // d_beta_sym[c][f][s] = (d_f B_s^c + d_s B_f^c)/2
  for (int cc2 = 0; cc2 < 3; ++cc2)
    for (int f = 0; f < 3; ++f)
      for (int s = 0; s < 3; ++s) dbs[cc2][f][s] = 0.5 * (dB[f][s][cc2] + dB[s][f][cc2]);
  double dmta[3], dac[3];
  for (int d = 0; d < 3; ++d) {
    double v = 0.0, w = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        v += hi[a][b] * dAt[d][a][b];
        w += dup[d][a][b] * At[a][b];
      }
    dmta[d] = v;
    dac[d] = w;
  }

  double hr[3][3], ar[3][3];
  for (int f = 0; f < 3; ++f)
    for (int s = 0; s < 3; ++s) {
      double v = 2.0 * (beta[0] * D[0][f][s] + beta[1] * D[1][f][s] + beta[2] * D[2][f][s]);
      double w = 0.0;
      for (int k = 0; k < 3; ++k) w += h[k][f] * B[s][k] + h[k][s] * B[f][k];
      v += w;
      v -= (2.0 / 3.0) * h[f][s] * divb;
      v -= 2.0 * alpha * (At[f][s] - (1.0 / 3.0) * h[f][s] * trA);
      v -= tau * (det - 1.0) * h[f][s];
      hr[f][s] = v;

      const double source = chi * (-hess[f][s] + alpha * rz[f][s]);
      const double trace_source = -lap_hess + alpha * scalar;
      double a = source - (h[f][s] / 3.0) * trace_source;
      double y = 0.0;
      for (int k = 0; k < 3; ++k) y += At[k][f] * B[s][k] + At[k][s] * B[f][k];
      a += y;
      a -= (2.0 / 3.0) * At[f][s] * divb;
      a += alpha * At[f][s] * (K - 2.0 * cc_ * Th);
      double z = 0.0;
      for (int l = 0; l < 3; ++l) {
        double m = 0.0;
        for (int n = 0; n < 3; ++n) m += hi[l][n] * At[n][s];
        z += At[f][l] * m;
      }
      a -= 2.0 * alpha * z;
      a -= tau * h[f][s] * trA;
      a += beta[0] * dAt[0][f][s] + beta[1] * dAt[1][f][s] + beta[2] * dAt[2][f][s];
      ar[f][s] = a;
    }

  double ln_lapse = beta[0] * A[0] + beta[1] * A[1] + beta[2] * A[2];
  ln_lapse -= G * (K - c.k0 - 2.0 * cc_ * Th);
  double shift_r[3];
  for (int a = 0; a < 3; ++a)
    shift_r[a] = sS * ((beta[0] * B[0][a] + beta[1] * B[1][a] + beta[2] * B[2][a]) + shift_f * bb[a]);
  double ln_phi = beta[0] * P[0] + beta[1] * P[1] + beta[2] * P[2];
  ln_phi += (alpha * K - divb) / 3.0;

  double kr = beta[0] * dK[0] + beta[1] * dK[1] + beta[2] * dK[2] - lap_hess;
  kr += alpha * scalar + alpha * K * (K - 2.0 * cc_ * Th);
  kr -= 3.0 * alpha * c.kappa1 * (1.0 + c.kappa2) * Th;
  double aa = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) aa += At[a][b] * aup[a][b];
  const double cs2 = c.cleaning_speed * c.cleaning_speed;
  double thr = beta[0] * dTh[0] + beta[1] * dTh[1] + beta[2] * dTh[2];
  thr += 0.5 * alpha * cs2 * scalar;
  thr += 0.5 * alpha * cs2 * (-aa + (2.0 / 3.0) * K * K);
  thr -= alpha * cc_ * Th * K;
  thr -= alpha * (zu[0] * A[0] + zu[1] * A[1] + zu[2] * A[2]);
  thr -= alpha * c.kappa1 * (2.0 + c.kappa2) * Th;
  // Diagnostic stores use the same local inputs as the unmodified Theta update.
  // They do not replace its sequential accumulation or alter any evolved slot.
  if (theta_terms) {
    const double values[12] = {
      beta[0] * dTh[0] + beta[1] * dTh[1] + beta[2] * dTh[2],
      0.5 * alpha * cs2 * scalar,
      0.5 * alpha * cs2 * (-aa + (2.0 / 3.0) * K * K),
      -alpha * cc_ * Th * K,
      -alpha * (zu[0] * A[0] + zu[1] * A[1] + zu[2] * A[2]),
      -alpha * c.kappa1 * (2.0 + c.kappa2) * Th,
      scalar, aa, zu[0] * A[0] + zu[1] * A[1] + zu[2] * A[2],
      alpha, chi, thr
    };
    for (int term = 0; term < 12; ++term) theta_terms[term * points + at] = values[term];
  }

  double gr[3], gnon[3];
  for (int cp = 0; cp < 3; ++cp) {
    const double adv = beta[0] * dGh[0][cp] + beta[1] * dGh[1][cp] + beta[2] * dGh[2][cp];
    double v = adv;
    double t;
    t = 0.0; for (int x = 0; x < 3; ++x) t += hi[cp][x] * dK[x];
    v -= (4.0 / 3.0) * alpha * t;
    t = 0.0; for (int x = 0; x < 3; ++x) t += hi[cp][x] * dTh[x];
    v += 2.0 * alpha * t;
    t = 0.0; for (int f = 0; f < 3; ++f) for (int s = 0; s < 3; ++s) t += hi[f][s] * dbs[cp][f][s];
    v += sS * t;
    t = 0.0; for (int f = 0; f < 3; ++f) for (int s = 0; s < 3; ++s) t += hi[cp][f] * dbs[s][f][s];
    v += sS * (1.0 / 3.0) * t;
    t = 0.0; for (int x = 0; x < 3; ++x) t += hi[cp][x] * dmta[x];
    v += sS * 2.0 * alpha * t;
    v += (2.0 / 3.0) * tg[cp] * divb;
    t = 0.0; for (int x = 0; x < 3; ++x) t += tg[x] * B[x][cp];
    v -= t;
    double t1 = 0.0, t2 = 0.0;
    for (int f = 0; f < 3; ++f) for (int s = 0; s < 3; ++s) t1 += cc[cp][f][s] * aup[f][s];
    for (int x = 0; x < 3; ++x) t2 += aup[cp][x] * P[x];
    v += 2.0 * alpha * (t1 - 3.0 * t2);
    t = 0.0; for (int x = 0; x < 3; ++x) t += hi[x][cp] * (Th * A[x] + (2.0 / 3.0) * K * zl[x]);
    v -= 2.0 * alpha * t;
    t = 0.0; for (int x = 0; x < 3; ++x) t += aup[cp][x] * alpha * A[x];
    v -= 2.0 * t;
    t = 0.0; for (int x = 0; x < 3; ++x) t += hi[cp][x] * dac[x];
    v -= 4.0 * sS * alpha * t;
    double hz = 0.0; for (int x = 0; x < 3; ++x) hz += hi[cp][x] * zl[x];
    double hzb = 0.0; for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x) hzb += hi[y][x] * zl[y] * B[x][cp];
    v += 2.0 * c.kappa3 * ((2.0 / 3.0) * hz * divb - hzb);
    v -= 2.0 * alpha * c.kappa1 * hz;
    gr[cp] = v;
    gnon[cp] = v - adv;
  }

  double auxA[3];
  for (int d = 0; d < 3; ++d) {
    double v = beta[0] * dA[0][d] + beta[1] * dA[1][d] + beta[2] * dA[2][d];
    v -= G * (dK[d] - 2.0 * cc_ * dTh[d]);
    v -= sS * G * dmta[d];
    v += 2.0 * sS * G * dac[d];
    v += B[d][0] * A[0] + B[d][1] * A[1] + B[d][2] * A[2];
    auxA[d] = v;
  }

  double auxB[3][3];
  for (int d = 0; d < 3; ++d)
    for (int cp = 0; cp < 3; ++cp) {
      double v = sS * (beta[0] * dB[0][d][cp] + beta[1] * dB[1][d][cp] + beta[2] * dB[2][d][cp]);
      v += sS * shift_f * dbb[d][cp];
      if (profile) v += sS * shift_f * (A[d] + 2.0 * P[d]) * bb[cp];
      double curl_p = 0.0, curl_d = 0.0;
      for (int m = 0; m < 3; ++m) {
        curl_p += hi[cp][m] * (dP[d][m] - dP[m][d]);
        double dterm = 0.0;
        for (int u = 0; u < 3; ++u)
          for (int di = 0; di < 3; ++di) dterm += hi[u][di] * (dD[d][di][m][u] - dD[di][d][m][u]);
        curl_d += hi[cp][m] * dterm;
      }
      v += sS * alpha * alpha * ordering_mu * (curl_p - curl_d);
      v += sS * (B[d][0] * B[0][cp] + B[d][1] * B[1][cp] + B[d][2] * B[2][cp]);
      auxB[d][cp] = v;
    }

  double auxD[3][3][3];
  for (int d = 0; d < 3; ++d)
    for (int f = 0; f < 3; ++f)
      for (int s = 0; s < 3; ++s) {
        double v = beta[0] * dD[0][d][f][s] + beta[1] * dD[1][d][f][s] + beta[2] * dD[2][d][f][s];
        double t = 0.0;
        for (int k = 0; k < 3; ++k) t += 0.5 * (h[k][f] * dbs[k][d][s] + h[k][s] * dbs[k][d][f]);
        v += sS * t;
        v -= sS * (1.0 / 3.0) * h[f][s] * (dbs[0][d][0] + dbs[1][d][1] + dbs[2][d][2]);
        v -= alpha * dAt[d][f][s];
        v += (alpha / 3.0) * h[f][s] * dmta[d];
        double w = 0.0;
        for (int x = 0; x < 3; ++x) w += B[d][x] * D[x][f][s] + B[s][x] * D[d][x][f] + B[f][x] * D[d][x][s];
        v += w;
        v -= (2.0 / 3.0) * divb * D[d][f][s];
        v -= (2.0 / 3.0) * alpha * h[f][s] * dac[d];
        v -= alpha * A[d] * (At[f][s] - (1.0 / 3.0) * h[f][s] * trA);
        auxD[d][f][s] = v;
      }

  double auxP[3];
  for (int d = 0; d < 3; ++d) {
    double v = beta[0] * dP[0][d] + beta[1] * dP[1][d] + beta[2] * dP[2][d];
    v += (alpha / 3.0) * dK[d];
    v -= (sS / 3.0) * (dbs[0][d][0] + dbs[1][d][1] + dbs[2][d][2]);
    v += (sS * alpha / 3.0) * dmta[d];
    v += (alpha / 3.0) * A[d] * K;
    v += B[d][0] * P[0] + B[d][1] * P[1] + B[d][2] * P[2];
    v -= sS * (2.0 * alpha / 3.0) * dac[d];
    auxP[d] = v;
  }

  double br[3];
  for (int cp = 0; cp < 3; ++cp)
    br[cp] = sS * (gnon[cp] - c.shift_damping * bb[cp] + (beta[0] * dbb[0][cp] + beta[1] * dbb[1][cp] + beta[2] * dbb[2][cp]));

  // ---- write slots
  auto W = [&](int slot, double value) { out[slot * points + at] = value; };
  for (int s = 0; s < 6; ++s) {
    W(s, hr[SYM[s][0]][SYM[s][1]]);
    W(11 + s, ar[SYM[s][0]][SYM[s][1]]);
    for (int ax = 0; ax < 3; ++ax) W(37 + 6 * ax + s, auxD[ax][SYM[s][0]][SYM[s][1]]);
  }
  W(6, ln_lapse);
  W(10, ln_phi);
  W(17, kr);
  W(18, thr);
  for (int a = 0; a < 3; ++a) {
    W(7 + a, shift_r[a]); W(19 + a, gr[a]); W(22 + a, br[a]); W(25 + a, auxA[a]); W(55 + a, auxP[a]);
    for (int b = 0; b < 3; ++b) W(28 + 3 * a + b, auxB[a][b]);
  }
  (void)sym_index;
}

}  // namespace

extern "C" {

// Returns 0 on success, nonzero on invalid input. params: lapse_coefficient, shift_coefficient,
// shift_damping, kappa1, kappa2, kappa3, cleaning_speed, ordering_mu, shift_strength,
// tau_inverse, k0, c, profile (0/1), shift_scale, ordering_scale (15 values).
// sbp_axes[a] = 1 selects SBP(4,2) (order 4 only) on axis a, 0 periodic.
static int evaluate_rhs(const double* q, double* out, int nx, int ny, int nz,
                const double* spacing, int order, const int* sbp_axes, const double* params,
                int threads, double* theta_terms) {
  if (!q || !out || !spacing || !params || !sbp_axes || nx < 5 || ny < 5 || nz < 5 || (order != 2 && order != 4)) return 1;
  Grid g{};
  g.nx = nx; g.ny = ny; g.nz = nz; g.order = order;
  g.n[0] = nx; g.n[1] = ny; g.n[2] = nz;
  for (int a = 0; a < 3; ++a) {
    if (!(spacing[a] > 0.0)) return 2;
    g.sbp[a] = sbp_axes[a] ? 1 : 0;
    if (g.sbp[a] && (order != 4 || g.n[a] < 9)) return 3;
    g.h[a] = spacing[a];
    g.inv[a] = 1.0 / ((order == 4 ? 12.0 : 2.0) * spacing[a]);
  }
  const Params c{params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                 params[8], params[9], params[10], params[11], params[12], params[13], params[14]};
  const std::int64_t total = static_cast<std::int64_t>(nx) * ny * nz;
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
  for (std::int64_t p = 0; p < total; ++p) {
    const int i = static_cast<int>(p / (static_cast<std::int64_t>(ny) * nz));
    const int j = static_cast<int>((p / nz) % ny);
    const int k = static_cast<int>(p % nz);
    point_rhs(q, out, g, c, i, j, k, theta_terms);
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int fo_ccz4_rhs(const double* q, double* out, int nx, int ny, int nz,
                const double* spacing, int order, const int* sbp_axes, const double* params, int threads) {
  return evaluate_rhs(q, out, nx, ny, nz, spacing, order, sbp_axes, params, threads, nullptr);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int fo_ccz4_rhs_theta_terms(const double* q, double* out, double* terms, int nx, int ny, int nz,
                const double* spacing, int order, const int* sbp_axes, const double* params, int threads) {
  if (!terms) return 1;
  return evaluate_rhs(q, out, nx, ny, nz, spacing, order, sbp_axes, params, threads, terms);
}

}  // extern "C"
