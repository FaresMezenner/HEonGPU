// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_ENCODING_H
#define HEONGPU_ENCODING_H

#include <curand_kernel.h>
#include "gpuntt/common/modular_arith.cuh"
#include "gpufft/complex.cuh"
#include <heongpu/util/bigintegerarith.cuh>

namespace heongpu
{

    __global__ void encode_kernel_bfv(Data64* message_encoded, Data64* message,
                                      Data64* location_info,
                                      Modulus64* plain_mod, int message_size);

    __global__ void decode_kernel_bfv(Data64* message, Data64* message_encoded,
                                      Data64* location_info);

    __global__ void encode_kernel_double_ckks_conversion(
        Data64* plaintext, double message, Modulus64* modulus,
        int coeff_modulus_count, double two_pow_64, int n_power);

    __global__ void encode_kernel_int_ckks_conversion(Data64* plaintext,
                                                      std::int64_t message,
                                                      Modulus64* modulus,
                                                      int n_power);

    __global__ void double_to_complex_kernel(double* input, Complex64* output);

    __global__ void complex_to_double_kernel(Complex64* input, double* output);

    __global__ void
    encode_kernel_ckks_conversion(Data64* plaintext, Complex64* complex_message,
                                  Modulus64* modulus, int coeff_modulus_count,
                                  double two_pow_64, int* reverse_order,
                                  int n_power);

    __global__ void encode_kernel_compose(
        Complex64* complex_message, Data64* plaintext, Modulus64* modulus,
        Data64* Mi_inv, Data64* Mi, Data64* upper_half_threshold,
        Data64* decryption_modulus, int coeff_modulus_count, double scale,
        double two_pow_64, int* reverse_order, int n_power);

    /**
     * @brief Kernel for extracting raw polynomial coefficients (INTT + CRT, NO FFT)
     * 
     * This kernel performs CRT reconstruction to get raw polynomial coefficients
     * without applying the special FFT (canonical embedding). This is essential
     * for R2L (RLWE-to-LWE) scheme switching where we need raw coefficient values
     * at specific positions, not slot values.
     * 
     * Key Differences from encode_kernel_compose:
     * - Processes N threads (all coefficients) instead of N/2 threads (slots)
     * - Does NOT apply bit-reversal permutation
     * - Outputs real coefficients, not complex slot values
     * 
     * Memory Layout:
     * plaintext[idx + (i << n_power)] contains coefficient idx under modulus i
     * 
     * @param output Raw polynomial coefficients (size N)
     * @param plaintext Plaintext data after INTT (RNS representation)
     * @param modulus RNS moduli
     * @param Mi_inv CRT inverse components
     * @param Mi CRT multiplier components
     * @param upper_half_threshold Threshold for centered representation
     * @param decryption_modulus Full CRT modulus
     * @param coeff_modulus_count Number of RNS moduli
     * @param scale CKKS scale factor (for division)
     * @param two_pow_64 Constant 2^64 for conversion
     * @param n_power log2(N) for indexing
     */
    __global__ void coefficient_compose_kernel(
        double* output, Data64* plaintext, Modulus64* modulus,
        Data64* Mi_inv, Data64* Mi, Data64* upper_half_threshold,
        Data64* decryption_modulus, int coeff_modulus_count, double scale,
        double two_pow_64, int n_power);

} // namespace heongpu
#endif // HEONGPU_ENCODING_H
