// kernels/09_high_reg_pressure.cu
extern "C" __global__
void high_reg_pressure(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float r0=0,r1=0,r2=0,r3=0,r4=0,r5=0,r6=0,r7=0,r8=0,r9=0;
        float r10=0,r11=0,r12=0,r13=0,r14=0,r15=0;
        for (int i=0;i<10;i++) {
            r0+=i; r1+=i; r2+=i; r3+=i; r4+=i; r5+=i; r6+=i; r7+=i; r8+=i; r9+=i;
            r10+=i; r11+=i; r12+=i; r13+=i; r14+=i; r15+=i;
        }
        data[idx] += r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
    }
}
