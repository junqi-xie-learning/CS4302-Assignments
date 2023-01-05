#include "tensor_core.h"

__half __float2half(const float &a) {
  // Convert floating-point numbers to half-precision floating-point numbers
  int a_rep = *(int *)&a;
  
  int sign = (a_rep >> 31) & 0x1;
  int exp = ((a_rep >> 23) & 0xff) - 127;
  int frac = a_rep & 0x7fffff;

  if (exp > 15) { // Infinity
    exp = 16;
    frac = 0;
  } else if (exp < -15) { // Denormalized
    exp = -15;
    frac = 0;
  }

  if (frac >> 12 & 0x1) { // Rounding to the nearest integer
    frac += 0x2000;
  }

  unsigned short x = (sign << 15) | ((exp + 15) << 10) | (frac >> 13);
  return __half{ x };
}

float __half2float(const __half &a) {
  // Convert half-precision floating-point numbers to floating-point numbers
  int sign = (a.__x >> 15) & 0x1;
  int exp = ((a.__x >> 10) & 0x1f) - 15;
  int frac = a.__x & 0x3ff;

  if (exp == 16) { // Infinity
    exp = 128;
  } else if (exp == -15 && frac == 0) { // Zero
    exp = -127;
  }

  int float_rep = (sign << 31) | ((exp + 127) << 23) | (frac << 13);
  return *(float *)&float_rep;
}

float operator*(const __half &lh, const __half &rh) {
  // Overloaded half-precision floating-point number multiplication
  int lh_sign = (lh.__x >> 15) & 0x1, rh_sign = (rh.__x >> 15) & 0x1;
  int lh_exp = ((lh.__x >> 10) & 0x1f) - 15, rh_exp = ((rh.__x >> 10) & 0x1f) - 15;
  int lh_frac = lh.__x & 0x3ff, rh_frac = rh.__x & 0x3ff;

  int sign = lh_sign ^ rh_sign;
  if (lh_exp == -15 && lh_frac == 0 || rh_exp == -15 && rh_frac == 0) { // Zero
    return 0;
  } else if (lh_exp == 16 || rh_exp == 16) { // Infinity
    return (sign << 31) | (0x7f800000);
  }

  int exp = lh_exp + rh_exp;
  int frac = ((lh_frac + 0x400) * (rh_frac + 0x400)) << 3;

  if (frac >> 24) {
    frac >>= 1;
    exp++;
  }

  int float_rep = (sign << 31) | ((exp + 127) << 23) | (frac & 0x7fffff);
  return *(float *)&float_rep;
}

GPU::GPU() {
  // Initialize GPU resources reasonably, including regfile size and global
  // memory size, assuming sm=1 and warp_num=1
  memory_ = new unsigned[TILE_WIDTH * TILE_WIDTH * 2]{ };
  regfile_ = new unsigned[WARP_SIZE_ * 256]{ };
  pregfile_ = new bool[WARP_SIZE_ * 8]{ };

  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    pregfile_[PT * WARP_SIZE_ + threadIdx] = true;
  }
}

GPU::~GPU() {
  // Release GPU resources
  delete[] memory_;
  delete[] regfile_;
  delete[] pregfile_;
}

void GPU::SIM_LDG_INSTR(unsigned sz, unsigned Rd, unsigned Ra, unsigned imm) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LDG implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned ra_idx = (ra_data + imm) / sizeof(unsigned);

    for (int i = 0; i < sz / (sizeof(unsigned) * 8); i++) {
      regfile_[(Rd + i) * WARP_SIZE_ + threadIdx] = memory_[ra_idx + i];
    }
  }
}

void GPU::SIM_STG_INSTR(unsigned sz, unsigned Ra, unsigned imm, unsigned Sb) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // STG implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned ra_idx = (ra_data + imm) / sizeof(unsigned);

    for (int i = 0; i < sz / (sizeof(unsigned) * 8); i++) {
      memory_[ra_idx + i] = regfile_[(Sb + i) * WARP_SIZE_ + threadIdx];
    }
  }
}

void GPU::load_ra_frag(__half *ra_data, int step, unsigned Ra, unsigned groupIdx) {
  // load ra_data for a thread group
  int ra_idx = groupIdx * GROUP_SIZE_;
  if (step == 1 || step == 3)
    ra_idx += GROUP_SIZE_ / 2;
  for (int i = 0; i < GROUP_SIZE_ / 2; i++) {
    for (int j = 0; j < GROUP_SIZE_ / 2; j++) {
      *(unsigned *)&ra_data[i * GROUP_SIZE_ + j * 2] =
        regfile_[(Ra + j) * WARP_SIZE_ + (ra_idx + i)];
    }
  }
}

void GPU::load_sb_frag(__half *sb_data, int step, unsigned Sb, unsigned groupIdx) {
  // load sb_data for a thread group
  if ((step == 0 || step == 1) && groupIdx >= GROUP_NUM_ / 2)
    groupIdx -= GROUP_NUM_ / 2;
  else if ((step == 2 || step == 3) && groupIdx < GROUP_NUM_ / 2)
    groupIdx += GROUP_NUM_ / 2;
  int sb_idx = groupIdx * GROUP_SIZE_;
  for (int i = 0; i < GROUP_SIZE_; i++) {
    for (int j = 0; j < GROUP_SIZE_ / 2; j++) {
      *(unsigned *)&sb_data[i * GROUP_SIZE_ + j * 2] =
        regfile_[(Sb + j) * WARP_SIZE_ + (sb_idx + i)];
    }
  }
}

void GPU::load_sc_frag(float *sc_data, unsigned Sc, unsigned groupIdx) {
  // load sc_data for a thread group
  int sc_idx = groupIdx * GROUP_SIZE_;
  for (int i = 0; i < GROUP_SIZE_ / 2; i++) {
    for (int j = 0; j < GROUP_SIZE_ / 2; j++) {
      *(unsigned *)&sc_data[i * GROUP_SIZE_ + j] =
        regfile_[(Sc + j) * WARP_SIZE_ + (sc_idx + i)];
      *(unsigned *)&sc_data[i * GROUP_SIZE_ + GROUP_SIZE_ / 2 + j] =
        regfile_[(Sc + j) * WARP_SIZE_ + (sc_idx + GROUP_SIZE_ / 2 + i)];
    }
  }
}

void GPU::frag_mult(float *rd_data, __half *ra_data, __half *sb_data, float *sc_data) {
  // fragment multiplication
  for (int i = 0; i < GROUP_SIZE_ / 2; i++) {
    for (int j = 0; j < GROUP_SIZE_; j++) {
      for (int k = 0; k < GROUP_SIZE_; k++) {
        rd_data[i * GROUP_SIZE_ + j] +=
          ra_data[i * GROUP_SIZE_ + k] * sb_data[j * GROUP_SIZE_ + k];
      }
      rd_data[i * GROUP_SIZE_ + j] += sc_data[i * GROUP_SIZE_ + j];
    }
  }
}

void GPU::store_rd_frag(float *rd_data, unsigned Rd, unsigned groupIdx) {
  // store rd_data for a thread group
  int rd_idx = groupIdx * GROUP_SIZE_;
  for (int i = 0; i < GROUP_SIZE_ / 2; i++) {
    for (int j = 0; j < GROUP_SIZE_ / 2; j++) {
      regfile_[(Rd + j) * WARP_SIZE_ + (rd_idx + i)] =
        *(unsigned *)&rd_data[i * GROUP_SIZE_ + j];
      regfile_[(Rd + j) * WARP_SIZE_ + (rd_idx + GROUP_SIZE_ / 2 + i)] =
        *(unsigned *)&rd_data[i * GROUP_SIZE_ + GROUP_SIZE_ / 2 + j];
    }
  }
}

void GPU::SIM_HMMA_INSTR_STEP0(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc) {
  // for: thread group execution
  for (int groupIdx = 0; groupIdx < GROUP_NUM_; groupIdx++) {
    // HMMA.STEP0 implementation
    __half ra_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
           sb_data[GROUP_SIZE_ * GROUP_SIZE_]{ };
    float sc_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
          rd_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ };

    load_ra_frag(ra_data, 0, Ra, groupIdx);
    load_sb_frag(sb_data, 0, Sb, groupIdx);
    load_sc_frag(sc_data, Sc, groupIdx);
    frag_mult(rd_data, ra_data, sb_data, sc_data);
    store_rd_frag(rd_data, Rd, groupIdx);
  }
}

void GPU::SIM_HMMA_INSTR_STEP1(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc) {
  // for: thread group execution
  for (int groupIdx = 0; groupIdx < GROUP_NUM_; groupIdx++) {
    // HMMA.STEP1 implementation
    __half ra_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
           sb_data[GROUP_SIZE_ * GROUP_SIZE_]{ };
    float sc_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
          rd_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ };

    load_ra_frag(ra_data, 1, Ra, groupIdx);
    load_sb_frag(sb_data, 1, Sb, groupIdx);
    load_sc_frag(sc_data, Sc, groupIdx);
    frag_mult(rd_data, ra_data, sb_data, sc_data);
    store_rd_frag(rd_data, Rd, groupIdx);
  }
}

void GPU::SIM_HMMA_INSTR_STEP2(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc) {
  // for: thread group execution
  for (int groupIdx = 0; groupIdx < GROUP_NUM_; groupIdx++) {
    // HMMA.STEP2 implementation
    __half ra_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
           sb_data[GROUP_SIZE_ * GROUP_SIZE_]{ };
    float sc_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
          rd_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ };
    
    load_ra_frag(ra_data, 2, Ra, groupIdx);
    load_sb_frag(sb_data, 2, Sb, groupIdx);
    load_sc_frag(sc_data, Sc, groupIdx);
    frag_mult(rd_data, ra_data, sb_data, sc_data);
    store_rd_frag(rd_data, Rd, groupIdx);
  }
}

void GPU::SIM_HMMA_INSTR_STEP3(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc) {
  // for: thread group execution
  for (int groupIdx = 0; groupIdx < GROUP_NUM_; groupIdx++) {
    // HMMA.STEP3 implementation
    __half ra_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
           sb_data[GROUP_SIZE_ * GROUP_SIZE_]{ };
    float sc_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ },
          rd_data[GROUP_SIZE_ / 2 * GROUP_SIZE_]{ };
    
    load_ra_frag(ra_data, 3, Ra, groupIdx);
    load_sb_frag(sb_data, 3, Sb, groupIdx);
    load_sc_frag(sc_data, Sc, groupIdx);
    frag_mult(rd_data, ra_data, sb_data, sc_data);
    store_rd_frag(rd_data, Rd, groupIdx);
  }
}

void GPU::SIM_MOV_INSTR(bool WIDE, unsigned Rd, unsigned long imm) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // MOV implementation
    regfile_[Rd * WARP_SIZE_ + threadIdx] = imm & 0xffffffff;
    if (WIDE)
      regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = (imm >> 32) & 0xffffffff;
  }
}

void GPU::SIM_IMAD_INSTR(bool WIDE, fmt_t fmt, unsigned Rd, unsigned Ra, unsigned Sb,
                         unsigned Sc, bool negP, bool negA) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // IMAD implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    if (!WIDE) {
      unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
      unsigned data = 0;
      if (fmt == U32) data = (negP ? -1 : 1) * ra_data * sb_data + 
                             (negA ? -1 : 1) * sc_data;
      else (int &)data = (negP ? -1 : 1) * (int)ra_data * (int)sb_data +
                         (negA ? -1 : 1) * (int)sc_data; 
      regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
    } else {
      uint64_t sc_data = (uint64_t)regfile_[Sc * WARP_SIZE_ + threadIdx];
      sc_data |= (uint64_t)regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx] << 32;
      uint64_t data = 0;
      if (fmt == U32) data = (negP ? -1L : 1L) * ra_data * sb_data +
                             (negA ? -1L : 1L) * sc_data;
      else (int64_t &)data = (negP ? -1L : 1L) * (int64_t)ra_data * (int64_t)sb_data +
                             (negA ? -1L : 1L) * (int64_t)sc_data;
      regfile_[Rd * WARP_SIZE_ + threadIdx] = data & 0xffffffff;
      regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = (data >> 32) & 0xffffffff;
    }
  }
}

void GPU::SIM_LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                         unsigned imm) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LOP3 implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = Sb; // Sb is used as an immediate value
    unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    unsigned data = 0;
    if (imm & 0x01) data |= (~ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x02) data |= (~ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x04) data |= (~ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x08) data |= (~ra_data) & (sb_data) & (sc_data);
    if (imm & 0x10) data |= (ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x20) data |= (ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x40) data |= (ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x80) data |= (ra_data) & (sb_data) & (sc_data);
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

void GPU::SIM_SHF_INSTR(shf_dir_t dir, fmt_t maxshift, bool HI, unsigned Rd,
                        unsigned Ra, unsigned Sb, unsigned Sc) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // SHF implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = Sb; // Sb is used as an immediate value
    unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    uint64_t data = 0;
    if (maxshift == U32) {
      data = (uint64_t)sc_data << 32 | ra_data;
      if (dir == SHF_L) data <<= (HI ? sb_data + 32 : sb_data);
      else data >>= (HI ? sb_data + 32 : sb_data);
    } else {
      (int64_t &)data = (int64_t)sc_data << 32 | (int)ra_data;
      if (dir == SHF_L) (int64_t &)data <<= (HI ? sb_data + 32 : sb_data);
      else (int64_t &)data >>= (HI ? sb_data + 32 : sb_data);
    }
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data & 0xffffffff;
  }
}

void GPU::SIM_S2R_INSTR(unsigned Rd, s_reg_t s_reg) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // S2R implementation
    switch (s_reg)
    {
    case SR_LAINID:
      regfile_[Rd * WARP_SIZE_ + threadIdx] = threadIdx;
      break;
    default:
      regfile_[Rd * WARP_SIZE_ + threadIdx] = 0;
      break;
    }
  }
}

void GPU::SIM_CS2R_INSTR(unsigned Rd, s_reg_t s_reg) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // CS2R implementation
    switch (s_reg)
    {
    case SRZ:
      regfile_[Rd * WARP_SIZE_ + threadIdx] = 0;
      break;
    default:
      break;
    }
  }
}

void GPU::SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                        unsigned Sc, unsigned imm, unsigned Pd0, unsigned Ps0) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LEA implementation
    unsigned ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    uint64_t sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    uint64_t data = ra_data;
    if (Sc != 255){
      data = (sc_data << 32) | ra_data;
    }
    if (HI)
      data = data >> (32 - imm);
    else
      data = data << imm;
    data += sb_data;
    if (X) data += pregfile_[Ps0 * WARP_SIZE_ + threadIdx];
    if (Pd0 != 7)
      pregfile_[Pd0 * WARP_SIZE_ + threadIdx] = ((data >> 32) & 0x1);
    data &= 0xffffffff;
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

void GPU::SIM_EXIT_INSTR() {
  // EXIT implementation
  memset(regfile_, 0, sizeof(unsigned) * WARP_SIZE_ * 256);
  memset(pregfile_, 0, sizeof(bool) * WARP_SIZE_ * 8);

  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    pregfile_[PT * WARP_SIZE_ + threadIdx] = true;
  }
}

void simMalloc(void **ptr, size_t size, GPU &volta) {
  // sim cudaMalloc
  // Request GPU memory
  *ptr = (void *)volta.allocated;
  volta.allocated += size;
}

void simMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind kind,
               GPU &volta) {
  // sim cudaMemcpy
  // memcpy host memory to class GPU memory or
  // memcpy class GPU memory to host memory
  switch (kind)
  {
  case MemcpyHostToDevice:
    dst = (void *)volta.memory_ + (unsigned long)dst;
    break;
  case MemcpyDeviceToHost:
    src = (void *)volta.memory_ + (unsigned long)src;
    break;
  default:
    break;
  }
  memcpy(dst, src, count);
}

void wmma_kernel(__half *a, __half *b, float *c, dim3 &gridDim,
                 dim3 &blockDim, GPU &volta) {
  // device kernel function
  // gridDim & blockDim
  // assume c[0x0][0x28]=0
  const int c_0_28 = 0;
  volta.SIM_MOV_INSTR(false, 1, c_0_28); // IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
  volta.SIM_S2R_INSTR(0, SR_LAINID); // S2R R0, SR_LANEID ;
  volta.SIM_MOV_INSTR(false, 28, 0x10); // IMAD.MOV.U32 R28, RZ, RZ, 0x10 ;
  volta.SIM_SHF_INSTR(SHF_R, U32, true, 4, RZ, 0x4, 0); // SHF.R.U32.HI R4, RZ, 0x4, R0.reuse ;
  volta.SIM_SHF_INSTR(SHF_R, U32, true, 2, RZ, 0x2, 0); // SHF.R.U32.HI R2, RZ, 0x2, R0 ;
  volta.SIM_LOP3_INSTR(0, 0, 0x3, RZ, 0xc0); // LOP3.LUT R0, R0, 0x3, RZ, 0xc0, !PT ;
  volta.SIM_SHF_INSTR(SHF_L, U32, false, 5, 4, 0x2, RZ); // IMAD.SHL.U32 R5, R4, 0x4, RZ ;
  volta.SIM_LOP3_INSTR(3, 2, 0x3, RZ, 0xc0); // LOP3.LUT R3, R2, 0x3, RZ, 0xc0, !PT ;
  volta.SIM_LOP3_INSTR(2, 5, 0x4, 0, 0xe2); // LOP3.LUT R2, R5, 0x4, R0, 0xe2, !PT ;
  volta.SIM_SHF_INSTR(SHF_L, U32, false, 8, 3, 0x3, RZ); // IMAD.SHL.U32 R8, R3.reuse, 0x8, RZ ;
  volta.SIM_SHF_INSTR(SHF_R, U32, true, 7, RZ, 0x1, 3); // SHF.R.U32.HI R7, RZ, 0x1, R3 ;
  volta.SIM_SHF_INSTR(SHF_L, U32, false, 9, 3, 0x3, RZ); // IMAD.SHL.U32 R9, R3, 0x8, RZ ;
  volta.SIM_LOP3_INSTR(6, 2, 0x2, RZ, 0xc0); // LOP3.LUT R6, R2.reuse, 0x2, RZ, 0xc0, !PT ;
  volta.SIM_MOV_INSTR(false, 3, 0x0); // IMAD.MOV.U32 R3, RZ, RZ, RZ ;
  volta.SIM_LOP3_INSTR(5, 2, 0x5, RZ, 0xc0); // LOP3.LUT R5, R2, 0x5, RZ, 0xc0, !PT ;
  volta.SIM_MOV_INSTR(false, Rimm, 0x8);
  volta.SIM_IMAD_INSTR(false, S32, 2, 7, Rimm, 6); // IMAD R2, R7.reuse, 0x8, R6 ;
  volta.SIM_LOP3_INSTR(5, 8, 0x8, 5, 0xe2); // LOP3.LUT R5, R8, 0x8, R5, 0xe2, !PT ;
  volta.SIM_LOP3_INSTR(6, 4, 0x1, RZ, 0xc0); // LOP3.LUT R6, R4, 0x1, RZ, 0xc0, !PT ;
  volta.SIM_LEA_INSTR(false, false, 7, 7, 0, RZ, 0x3); // LEA R7, R7, R0, 0x3 ;
  volta.SIM_MOV_INSTR(false, Rimm, 0x10);
  volta.SIM_IMAD_INSTR(true, U32, 4, 5, Rimm, 2); // IMAD.WIDE.U32 R4, R5, 0x10, R2 ;
  volta.SIM_LOP3_INSTR(3, 9, 0x8, 0, 0xe2); // LOP3.LUT R3, R9, 0x8, R0, 0xe2, !PT ;
  volta.SIM_MOV_INSTR(false, Rimm, 0x4);
  volta.SIM_IMAD_INSTR(false, S32, 7, 6, Rimm, 7); // IMAD R7, R6.reuse, 0x4, R7 ;
  volta.SIM_IMAD_INSTR(false, S32, 3, 6, Rimm, 3); // IMAD R3, R6, 0x4, R3 ;
  volta.SIM_MOV_INSTR(true, Rimm, (unsigned long)c);
  volta.SIM_LEA_INSTR(false, false, 2, 4, Rimm, RZ, 0x2, 0, 7); // LEA R2, P0, R4, c[0x0][0x170], 0x2 ;
  volta.SIM_SHF_INSTR(SHF_L, U32, false, 7, 7, 0x1, RZ); // IMAD.SHL.U32 R7, R7, 0x2, RZ ;
  volta.SIM_SHF_INSTR(SHF_L, U32, false, 20, 3, 0x1, RZ); // SHF.L.U32 R20, R3, 0x1, RZ ;
  volta.SIM_LEA_INSTR(true, true, 3, 4, Rimm + 1, 5, 0x2, 7, 0); // LEA.HI.X R3, R4, c[0x0][0x174], R5, 0x2, P0 ;
  volta.SIM_MOV_INSTR(true, Rimm, (unsigned long)a);
  volta.SIM_IMAD_INSTR(true, U32, 20, 20, 28, Rimm); // IMAD.WIDE.U32 R20, R20, R28, c[0x0][0x160] ;
  volta.SIM_MOV_INSTR(true, Rimm, (unsigned long)b);
  volta.SIM_IMAD_INSTR(true, U32, 28, 7, 28, Rimm); // IMAD.WIDE.U32 R28, R7, R28, c[0x0][0x168] ;
  volta.SIM_LDG_INSTR(64, 8, 2, 0x0); // LDG.E.64.SYS R8, [R2] ;
  volta.SIM_LDG_INSTR(64, 10, 2, 0x80); // LDG.E.64.SYS R10, [R2+0x80] ;
  volta.SIM_LDG_INSTR(128, 24, 20, 0x0); // LDG.E.128.SYS R24, [R20] ;
  volta.SIM_LDG_INSTR(128, 16, 28, 0x0); // LDG.E.128.SYS R16, [R28] ;
  volta.SIM_LDG_INSTR(64, 4, 2, 0x10); // LDG.E.64.SYS R4, [R2+0x10] ;
  volta.SIM_LDG_INSTR(64, 6, 2, 0x90); // LDG.E.64.SYS R6, [R2+0x90] ;
  volta.SIM_LDG_INSTR(128, 20, 20, 0x10); // LDG.E.128.SYS R20, [R20+0x10] ;
  volta.SIM_LDG_INSTR(128, 12, 28, 0x10); // LDG.E.128.SYS R12, [R28+0x10] ;
  volta.SIM_HMMA_INSTR_STEP0(8, 24, 16, 8); // HMMA.884.F32.F32.STEP0 R8, R24.reuse.ROW, R16.reuse.COL, R8 ;
  volta.SIM_HMMA_INSTR_STEP1(10, 24, 16, 10); // HMMA.884.F32.F32.STEP1 R10, R24.reuse.ROW, R16.reuse.COL, R10 ;
  volta.SIM_HMMA_INSTR_STEP2(4, 24, 16, 4); // HMMA.884.F32.F32.STEP2 R4, R24.reuse.ROW, R16.reuse.COL, R4 ;
  volta.SIM_HMMA_INSTR_STEP3(6, 24, 16, 6); // HMMA.884.F32.F32.STEP3 R6, R24.ROW, R16.COL, R6 ;
  volta.SIM_HMMA_INSTR_STEP0(8, 26, 18, 8); // HMMA.884.F32.F32.STEP0 R8, R26.reuse.ROW, R18.reuse.COL, R8 ;
  volta.SIM_HMMA_INSTR_STEP1(10, 26, 18, 10); // HMMA.884.F32.F32.STEP1 R10, R26.reuse.ROW, R18.reuse.COL, R10 ;
  volta.SIM_HMMA_INSTR_STEP2(4, 26, 18, 4); // HMMA.884.F32.F32.STEP2 R4, R26.reuse.ROW, R18.reuse.COL, R4 ;
  volta.SIM_HMMA_INSTR_STEP3(6, 26, 18, 6); // HMMA.884.F32.F32.STEP3 R6, R26.ROW, R18.COL, R6 ;
  volta.SIM_HMMA_INSTR_STEP0(8, 20, 12, 8); // HMMA.884.F32.F32.STEP0 R8, R20.reuse.ROW, R12.reuse.COL, R8 ;
  volta.SIM_HMMA_INSTR_STEP1(10, 20, 12, 10); // HMMA.884.F32.F32.STEP1 R10, R20.reuse.ROW, R12.reuse.COL, R10 ;
  volta.SIM_HMMA_INSTR_STEP2(4, 20, 12, 4); // HMMA.884.F32.F32.STEP2 R4, R20.reuse.ROW, R12.reuse.COL, R4 ;
  volta.SIM_HMMA_INSTR_STEP3(6, 20, 12, 6); // HMMA.884.F32.F32.STEP3 R6, R20.ROW, R12.COL, R6 ;
  volta.SIM_HMMA_INSTR_STEP0(8, 22, 14, 8); // HMMA.884.F32.F32.STEP0 R8, R22.reuse.ROW, R14.reuse.COL, R8 ;
  volta.SIM_HMMA_INSTR_STEP1(10, 22, 14, 10); // HMMA.884.F32.F32.STEP1 R10, R22.reuse.ROW, R14.reuse.COL, R10 ;
  volta.SIM_HMMA_INSTR_STEP2(4, 22, 14, 4); // HMMA.884.F32.F32.STEP2 R4, R22.reuse.ROW, R14.reuse.COL, R4 ;
  volta.SIM_HMMA_INSTR_STEP3(6, 22, 14, 6); // HMMA.884.F32.F32.STEP3 R6, R22.ROW, R14.COL, R6 ;
  volta.SIM_STG_INSTR(64, 2, 0x0, 8); // STG.E.64.SYS [R2], R8 ;
  volta.SIM_STG_INSTR(64, 2, 0x80, 10); // STG.E.64.SYS [R2+0x80], R10 ;
  volta.SIM_STG_INSTR(64, 2, 0x10, 4); // STG.E.64.SYS [R2+0x10], R4 ;
  volta.SIM_STG_INSTR(64, 2, 0x90, 6); // STG.E.64.SYS [R2+0x90], R6 ;
  volta.SIM_EXIT_INSTR(); // EXIT ;
}

void gemm(__half *a, __half *b, float *c, int m, int n, int k) {
  // host function gemm
  GPU volta;
  dim3 blockDim(WARP_SIZE, 1, 1), gridDim(1, 1, 1);

  __half *d_a, *d_b;
  float* d_c;
  simMalloc((void**)(&d_a), sizeof(__half) * TILE_WIDTH * TILE_WIDTH, volta);
  simMalloc((void**)(&d_b), sizeof(__half) * TILE_WIDTH * TILE_WIDTH, volta);
  simMalloc((void**)(&d_c), sizeof(float) * TILE_WIDTH * TILE_WIDTH, volta);

  int m_tile = ceil(m / (float)TILE_WIDTH), n_tile = ceil(n / (float)TILE_WIDTH),
      k_tile = ceil(k / (float)TILE_WIDTH);

  for (int i = 0; i < m_tile; i++) {
    for (int j = 0; j < k_tile; j++) {
      for (int l = 0; l < n_tile; l++) {
        int m_ = min(TILE_WIDTH, m - i * TILE_WIDTH);
        int k_ = min(TILE_WIDTH, k - j * TILE_WIDTH);
        int n_ = min(TILE_WIDTH, n - l * TILE_WIDTH);

        for (int ii = 0; ii < m_; ii++)
        {
          simMemcpy(&d_a[ii * TILE_WIDTH], &a[(i * TILE_WIDTH + ii) * n + l * TILE_WIDTH],
                    sizeof(__half) * n_, MemcpyHostToDevice, volta);
        }
        for (int jj = 0; jj < k_; jj++)
        {
          simMemcpy(&d_b[jj * TILE_WIDTH], &b[(j * TILE_WIDTH + jj) * n + l * TILE_WIDTH],
                    sizeof(__half) * n_, MemcpyHostToDevice, volta);
        }
        for (int ii = 0; ii < m_; ii++)
        {
          simMemcpy(&d_c[ii * TILE_WIDTH], &c[(i * TILE_WIDTH + ii) * k + j * TILE_WIDTH],
                    sizeof(float) * k_, MemcpyHostToDevice, volta);
        }
        wmma_kernel(d_a, d_b, d_c, gridDim, blockDim, volta);
        for (int ii = 0; ii < m_; ii++)
        {
          simMemcpy(&c[(i * TILE_WIDTH + ii) * k + j * TILE_WIDTH], &d_c[ii * TILE_WIDTH],
                    sizeof(float) * k_, MemcpyDeviceToHost, volta);
        }
        memset(volta.memory_, 0, sizeof(unsigned) * TILE_WIDTH * TILE_WIDTH * 2);
      }
    }
  }
}
