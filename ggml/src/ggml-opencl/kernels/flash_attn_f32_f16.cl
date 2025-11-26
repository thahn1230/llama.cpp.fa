#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

// 매크로가 정의되어 있지 않을 경우를 대비한 안전장치 (보통 컴파일 옵션으로 넘어옴)
#ifndef DK
#define DK 128
#endif

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)

// Decoding Kernel을 위한 전용 Wave Size (Adreno 최적화: 64)
#define DEC_WG_SIZE 64

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}

// =================================================================================================
// 1. Prefill Kernel (원본 유지)
// =================================================================================================
__kernel void flash_attn_f32_f16(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) {
            continue;
        }

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_acc0 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j][k]), dot_acc0);
                dot_acc1 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j+1][k]), dot_acc1);
            }
            ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
            ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

            if (is_causal) {
                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
            }

            if (k_row0 >= n_kv) score0 = -INFINITY;
            if (k_row1 >= n_kv) score1 = -INFINITY;

            if (mask_base != NULL) {
                const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
            }

            if (logit_softcap > 0.0f) {
                score0 = logit_softcap * tanh(score0 / logit_softcap);
                score1 = logit_softcap * tanh(score1 / logit_softcap);
            }

            const ACC_TYPE m_new = max(m_i, max(score0, score1));
            const ACC_TYPE p0 = exp(score0 - m_new);
            const ACC_TYPE p1 = exp(score1 - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_KV_ACC4(l_v[j][i]) + p1 * CONVERT_KV_ACC4(l_v[j+1][i]);
            }
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
}

// =================================================================================================
// 2. Optimized Decoding Kernel (Dimension Parallelism for Adreno)
// =================================================================================================
__kernel void flash_attn_f32_f16_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    // Q를 Shared Memory에 캐싱 (Half precision으로 변환하여 저장)
    // DK는 128 같은 매크로 상수로 정의되어야 함.
    __local half l_q[DK];

    const int tid = get_local_id(0); // 0 ~ 63
    const int head_batch_idx = get_global_id(1);
    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;
    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    // [1] Load Q: 64개 스레드가 협력하여 DK(128) 크기의 Q를 로드
    const global char* q_base = (const global char*)q_void + q_offset;
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global float* q_ptr = (const global float*)(q_base + q_row_offset);

    // DEC_WG_SIZE(64) 스트라이드로 로드
    for (int i = tid; i < DK; i += DEC_WG_SIZE) {
        l_q[i] = (half)q_ptr[i]; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // [2] Accumulator 초기화
    // 레지스터 스필 방지를 위해 각 스레드는 오직 2개의 Output(float2)만 관리
    // DK=128, WG=64 -> 1 thread covers 2 elements (index: tid*2, tid*2+1)
    float2 my_o_acc = (float2)(0.0f, 0.0f);
    
    float m_local = -INFINITY;
    float l_local = 0.0f;

    const int my_dim_base = tid * 2; // 내 스레드가 담당할 차원 시작점

    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    
    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
    
    // Sink 초기화
    if (sinks_void != NULL) {
        const global float* sinks_ptr = (const global float*)((const global char*)sinks_void + sinks_offset);
        m_local = sinks_ptr[head_idx];
    }

    // [3] Main Loop (Iterate over KV tokens)
    // 스레드당 연산량을 최소화하여 GPU 점유율 극대화
    for (int k_idx = 0; k_idx < n_kv; ++k_idx) {
        // A. Partial Dot Product (내 담당 차원만 계산)
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        
        // vload_half2: 2개의 half를 읽어 float2로 변환 (효율적)
        // k_ptr이 half*라고 가정 (KV_DATA_TYPE4가 half4이므로)
        const global half* k_ptr_half = (const global half*)(k_base + k_row_offset);
        
        // 범위 체크 (DK가 128이 아닐 경우 대비)
        float my_score_part = 0.0f;
        if (my_dim_base < DK) {
             // Local Memory Q 읽기 (half* -> half2 -> float2)
             half2 q_h2 = *(__local half2*)&l_q[my_dim_base];
             float2 q_val = convert_float2(q_h2);
             
             // Global Memory K 읽기
             float2 k_val = vload_half2(0, k_ptr_half + my_dim_base);
             
             my_score_part = dot(q_val, k_val);
        }

        // B. Reduction (Subgroup Sum) -> 전체 차원(128)에 대한 Score 완성
        // Adreno 하드웨어 레벨 리덕션 (매우 빠름)
        float score = sub_group_reduce_add(my_score_part);
        
        // Score Scaling & Masking (모든 스레드가 동일한 값 보유)
        score *= scale;
        if (mask_base != NULL) {
            const global half* mask_ptr = (const global half*)(mask_base);
            score += slope * (float)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // C. Online Softmax Update
        float m_prev = m_local;
        m_local = max(m_prev, score);
        
        float p = 0.0f;
        float scale_prev = 1.0f;
        if (m_local > -INFINITY) {
            p = exp(score - m_local);
            scale_prev = (m_prev > -INFINITY) ? exp(m_prev - m_local) : 0.0f;
        }

        l_local = l_local * scale_prev + p;

        // D. Accumulate V (Dimension Parallel)
        // 내 담당 차원(2개)에 해당하는 V값만 업데이트
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global half* v_ptr_half = (const global half*)(v_base + v_row_offset);
        
        if (my_dim_base < DK) {
            float2 v_val = vload_half2(0, v_ptr_half + my_dim_base);
            // my_o = my_o * scale + p * v
            my_o_acc = mad((float2)(p), v_val, my_o_acc * scale_prev);
        }
    }

    // [4] Final Normalize & Write
    if (l_local > 0.0f) {
        float l_inv = 1.0f / l_local;
        my_o_acc *= l_inv;
    } else {
        my_o_acc = (float2)(0.0f);
    }

    // Global Memory Write (각 스레드가 2개씩 씀)
    global char* o_base = (global char*)o_void + o_offset;
    ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global float* o_ptr = (global float*)(o_base + o_row_offset);

    if (my_dim_base < DK) {
        // float2로 기록하는 것이 대역폭에 유리할 수 있으나,
        // o_ptr이 float* 이므로 개별 기록 (컴파일러가 병합 최적화 수행함)
        o_ptr[my_dim_base] = my_o_acc.x;
        if (my_dim_base + 1 < DK) {
            o_ptr[my_dim_base + 1] = my_o_acc.y;
        }
    }
}