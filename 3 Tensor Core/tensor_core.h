#ifndef _TENSOR_CORE_H_
#define _TENSOR_CORE_H_
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
using namespace std;

enum cudaMemcpyKind {
  MemcpyHostToDevice = 0, /**< Host   -> Device */
  MemcpyDeviceToHost = 1  /**< Device -> Host */
};

enum fmt_t { U32 = 0, S32 };
enum shf_dir_t { SHF_L = 0, SHF_R };
enum s_reg_t { SRZ = 0, SR_LAINID, SR_TID_X, SR_TID_Y, SR_CTAID_X, SR_CTAID_Y };

struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
};

constexpr unsigned Rimm = 253;
constexpr unsigned RZ = 255;
constexpr unsigned PT = 7;

constexpr unsigned TILE_WIDTH = 16;
constexpr unsigned WARP_SIZE = 32;

struct __half {
  unsigned short __x;
  __half(unsigned short x = 0) : __x{ x } { }
};

extern __half __float2half(const float &a);

extern float __half2float(const __half &a);

extern float operator*(const __half &lh, const __half &rh);

class GPU {
public:
  GPU();
  ~GPU();
  void SIM_LDG_INSTR(unsigned sz, unsigned Rd, unsigned Ra, unsigned imm = 0x0);
  void SIM_STG_INSTR(unsigned sz, unsigned Ra, unsigned Sb, unsigned imm = 0x0);
  void SIM_HMMA_INSTR_STEP0(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc);
  void SIM_HMMA_INSTR_STEP1(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc);
  void SIM_HMMA_INSTR_STEP2(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc);
  void SIM_HMMA_INSTR_STEP3(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc);
  void SIM_MOV_INSTR(bool WIDE, unsigned Rd, unsigned long imm);
  void SIM_IMAD_INSTR(bool WIDE, fmt_t fmt, unsigned Rd, unsigned Ra, unsigned Sb,
                      unsigned Sc, bool negP = false, bool negA = false);
  void SIM_LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                      unsigned imm);
  void SIM_SHF_INSTR(shf_dir_t dir, fmt_t maxshift, bool HI, unsigned Rd,
                     unsigned Ra, unsigned Sb, unsigned Sc);
  void SIM_S2R_INSTR(unsigned Rd, s_reg_t s_reg);
  void SIM_CS2R_INSTR(unsigned Rd, s_reg_t s_reg);
  void SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                     unsigned Sc, unsigned imm, unsigned Pd0 = 7, unsigned Ps0 = 7);
  void SIM_EXIT_INSTR();
  unsigned *memory_;
  unsigned long allocated = 0;

private:
  // unsigned warpNum_;
  unsigned *regfile_;
  bool *pregfile_;
  const unsigned WARP_SIZE_ = 32;

  const unsigned GROUP_SIZE_ = 4;
  const unsigned GROUP_NUM_ = WARP_SIZE_ / GROUP_SIZE_;
  void load_ra_frag(__half *ra_data, int step, unsigned Ra, unsigned groupIdx);
  void load_sb_frag(__half *sb_data, int step, unsigned Sb, unsigned groupIdx);
  void load_sc_frag(float *sc_data, unsigned Sc, unsigned groupIdx);
  void frag_mult(float *rd_data, __half *ra_data, __half *sb_data, float *sc_data);
  void store_rd_frag(float *rd_data, unsigned Rd, unsigned groupIdx);
};

extern void simMalloc(void **ptr, size_t size, GPU &volta);

extern void simMemcpy(void *dst, void *src, size_t count,
                      enum cudaMemcpyKind kind, GPU &volta);

extern void wmma_kernel(__half *a, __half *b, float *c, dim3 &gridDim,
                        dim3 &blockDim, GPU &volta);

extern void gemm(__half *a, __half *b, float *c, int m, int n, int k);

#endif
