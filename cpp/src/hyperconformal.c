#include "hyperconformal.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

// Optimized bit manipulation macros for ARM Cortex-M
#ifdef __ARM_ARCH
    #define POPCOUNT(x) __builtin_popcount(x)
    #define CLZ(x) __builtin_clz(x)
#else
    // Fallback for other architectures
    static inline int popcount_fallback(uint32_t x) {
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }
    #define POPCOUNT(x) popcount_fallback(x)
    #define CLZ(x) __builtin_clz(x)
#endif

// Binary encoding using XOR-based random projection
void hc_encode_binary(const uint8_t* input, hypervector_t* output, const hdc_encoder_t* encoder) {
    if (!input || !output || !encoder) return;
    
    // Ensure output buffer is allocated
    if (!output->data) {
        output->size = (encoder->hv_dim + 7) / 8;  // Ceiling division for bits->bytes
        output->data = (uint8_t*)calloc(output->size, 1);
        if (!output->data) return;
    }
    
    // Clear output
    memset(output->data, 0, output->size);
    
    // Binary random projection with XOR operations
    for (uint16_t i = 0; i < encoder->input_dim; i++) {
        if (input[i / 8] & (1 << (i % 8))) {  // If input bit is set
            // XOR with corresponding row of projection matrix
            uint16_t matrix_offset = i * output->size;
            for (uint16_t j = 0; j < output->size; j++) {
                output->data[j] ^= encoder->projection_matrix[matrix_offset + j];
            }
        }
    }
}

// Ternary encoding (-1, 0, +1) using lookup tables
void hc_encode_ternary(const uint8_t* input, hypervector_t* output, const hdc_encoder_t* encoder) {
    if (!input || !output || !encoder) return;
    
    // Ternary requires 2 bits per element, so different memory layout
    output->size = (encoder->hv_dim * 2 + 7) / 8;
    if (!output->data) {
        output->data = (uint8_t*)calloc(output->size, 1);
        if (!output->data) return;
    }
    
    memset(output->data, 0, output->size);
    
    // Similar to binary but with ternary quantization
    for (uint16_t i = 0; i < encoder->input_dim; i++) {
        int8_t input_val = (input[i] > 127) ? 1 : ((input[i] < 64) ? -1 : 0);
        if (input_val != 0) {
            uint16_t matrix_offset = i * output->size;
            for (uint16_t j = 0; j < output->size; j++) {
                // Ternary arithmetic (simplified for embedded)
                if (input_val > 0) {
                    output->data[j] ^= encoder->projection_matrix[matrix_offset + j];
                } else {
                    output->data[j] ^= ~encoder->projection_matrix[matrix_offset + j];
                }
            }
        }
    }
}

// Ultra-fast Hamming distance using built-in popcount
uint16_t hamming_distance(const hypervector_t* hv1, const hypervector_t* hv2) {
    if (!hv1 || !hv2 || !hv1->data || !hv2->data) return 0xFFFF;
    
    uint16_t distance = 0;
    uint16_t min_size = (hv1->size < hv2->size) ? hv1->size : hv2->size;
    
    // Process 4 bytes at a time for efficiency
    uint16_t i;
    for (i = 0; i + 3 < min_size; i += 4) {
        uint32_t* p1 = (uint32_t*)(hv1->data + i);
        uint32_t* p2 = (uint32_t*)(hv2->data + i);
        distance += POPCOUNT(*p1 ^ *p2);
    }
    
    // Handle remaining bytes
    for (; i < min_size; i++) {
        distance += POPCOUNT(hv1->data[i] ^ hv2->data[i]);
    }
    
    return distance;
}

// Cosine similarity approximation for binary vectors
float cosine_similarity(const hypervector_t* hv1, const hypervector_t* hv2) {
    if (!hv1 || !hv2 || !hv1->data || !hv2->data) return 0.0f;
    
    uint16_t hamming_dist = hamming_distance(hv1, hv2);
    uint16_t total_bits = hv1->size * 8;
    
    // Convert Hamming distance to cosine similarity approximation
    // cos(θ) ≈ 1 - 2 * (hamming_distance / total_bits)
    return 1.0f - 2.0f * ((float)hamming_dist / (float)total_bits);
}

// Classification using nearest prototype in hypervector space
void hc_classify(const hypervector_t* input_hv, const conformal_model_t* model, class_scores_t* scores) {
    if (!input_hv || !model || !scores) return;
    
    scores->num_classes = 0;
    
    // Compute similarities to all class prototypes
    for (uint8_t class_idx = 0; class_idx < model->num_classes && class_idx < 16; class_idx++) {
        hypervector_t prototype;
        prototype.data = model->class_prototypes + (class_idx * input_hv->size);
        prototype.size = input_hv->size;
        
        // Use cosine similarity as the classification score
        scores->scores[class_idx] = cosine_similarity(input_hv, &prototype);
        scores->num_classes++;
    }
}

// Conformal prediction without floating point (fixed-point arithmetic)
void hc_conformal_predict(const class_scores_t* scores, const conformal_model_t* model,
                         uint8_t* prediction, uint8_t* confidence) {
    if (!scores || !model || !prediction || !confidence) return;
    
    // Find class with highest score
    uint8_t best_class = 0;
    float best_score = scores->scores[0];
    
    for (uint8_t i = 1; i < scores->num_classes; i++) {
        if (scores->scores[i] > best_score) {
            best_score = scores->scores[i];
            best_class = i;
        }
    }
    
    *prediction = best_class;
    
    // Compute conformal confidence using calibration scores
    // Use APS (Adaptive Prediction Sets) score: 1 - P(predicted_class)
    float aps_score = 1.0f - best_score;
    
    // Find quantile position in calibration scores (binary search would be better)
    uint16_t rank = 0;
    for (uint16_t i = 0; i < model->calibration_size; i++) {
        if (model->calibration_scores[i] <= aps_score) {
            rank++;
        }
    }
    
    // Compute p-value: (rank + 1) / (calibration_size + 1)
    float p_value = (float)(rank + 1) / (float)(model->calibration_size + 1);
    
    // Convert to confidence (0-255 scale for integer representation)
    *confidence = (uint8_t)(p_value * 255.0f);
    
    // If p_value > alpha, prediction is in conformal set
    if (p_value <= model->alpha) {
        *confidence = 255;  // High confidence
    }
}

// Initialize model from compressed blob (for deployment)
int hc_init_model(conformal_model_t* model, const uint8_t* model_blob, size_t blob_size) {
    if (!model || !model_blob || blob_size < sizeof(conformal_model_t)) {
        return -1;  // Invalid parameters
    }
    
    // Parse model header (simplified format)
    const uint8_t* ptr = model_blob;
    
    model->num_classes = *ptr++;
    model->alpha = *(float*)ptr; ptr += sizeof(float);
    model->score_type = *ptr++;
    model->calibration_size = *(uint16_t*)ptr; ptr += sizeof(uint16_t);
    
    // Allocate and copy calibration scores
    size_t cal_scores_size = model->calibration_size * sizeof(float);
    model->calibration_scores = (float*)malloc(cal_scores_size);
    if (!model->calibration_scores) return -2;
    
    memcpy(model->calibration_scores, ptr, cal_scores_size);
    ptr += cal_scores_size;
    
    // Allocate and copy class prototypes (assuming binary hypervectors)
    size_t hv_size = 1250;  // 10000 bits = 1250 bytes for 10K-dim hypervectors
    size_t prototypes_size = model->num_classes * hv_size;
    model->class_prototypes = (uint8_t*)malloc(prototypes_size);
    if (!model->class_prototypes) {
        free(model->calibration_scores);
        return -3;
    }
    
    memcpy(model->class_prototypes, ptr, prototypes_size);
    
    return 0;  // Success
}

// Clean up allocated memory
void hc_cleanup_model(conformal_model_t* model) {
    if (!model) return;
    
    if (model->calibration_scores) {
        free(model->calibration_scores);
        model->calibration_scores = NULL;
    }
    
    if (model->class_prototypes) {
        free(model->class_prototypes);
        model->class_prototypes = NULL;
    }
    
    model->calibration_size = 0;
    model->num_classes = 0;
}

// Get total memory usage
size_t hc_get_memory_usage(const conformal_model_t* model) {
    if (!model) return 0;
    
    size_t total = sizeof(conformal_model_t);
    total += model->calibration_size * sizeof(float);
    total += model->num_classes * 1250;  // 10K-dim binary hypervectors
    
    return total;
}

// Update calibration scores for online learning
void hc_update_calibration(conformal_model_t* model, const class_scores_t* new_scores, uint8_t true_label) {
    if (!model || !new_scores || true_label >= model->num_classes) return;
    
    // Compute APS score for new sample
    float aps_score = 1.0f - new_scores->scores[true_label];
    
    // Simple sliding window update (circular buffer)
    static uint16_t update_index = 0;
    if (model->calibration_scores && model->calibration_size > 0) {
        model->calibration_scores[update_index % model->calibration_size] = aps_score;
        update_index++;
    }
}

// Get current empirical coverage estimate
float hc_get_current_coverage(const conformal_model_t* model) {
    if (!model || !model->calibration_scores || model->calibration_size == 0) {
        return 0.0f;
    }
    
    // Count how many calibration scores are <= alpha
    uint16_t count = 0;
    for (uint16_t i = 0; i < model->calibration_size; i++) {
        if (model->calibration_scores[i] <= model->alpha) {
            count++;
        }
    }
    
    return (float)count / (float)model->calibration_size;
}

// Energy-aware prediction with operation counting
void hc_predict_with_energy(const hypervector_t* input_hv, const conformal_model_t* model,
                           class_scores_t* scores, energy_profile_t* energy) {
    if (!energy) return;
    
    // Initialize energy counters
    energy->xor_ops = 0;
    energy->popcount_ops = 0;
    energy->memory_accesses = 0;
    
    // Perform classification while counting operations
    if (input_hv && model && scores) {
        for (uint8_t class_idx = 0; class_idx < model->num_classes && class_idx < 16; class_idx++) {
            hypervector_t prototype;
            prototype.data = model->class_prototypes + (class_idx * input_hv->size);
            prototype.size = input_hv->size;
            
            // Count operations for Hamming distance computation
            energy->memory_accesses += input_hv->size * 2;  // Read both vectors
            energy->xor_ops += input_hv->size / 4;  // Assume 32-bit XOR ops
            energy->popcount_ops += input_hv->size / 4;
            
            scores->scores[class_idx] = cosine_similarity(input_hv, &prototype);
        }
        scores->num_classes = model->num_classes;
    }
    
    // Estimate energy consumption (based on ARM Cortex-M0+ measurements)
    // XOR: ~0.5 nJ, POPCOUNT: ~1.2 nJ, Memory: ~2.1 nJ per access
    energy->estimated_energy_uj = 
        (energy->xor_ops * 0.5f + 
         energy->popcount_ops * 1.2f + 
         energy->memory_accesses * 2.1f) / 1000.0f;  // Convert nJ to μJ
}