#ifndef HYPERCONFORMAL_H
#define HYPERCONFORMAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Core data structures for embedded HDC with conformal prediction
typedef struct {
    uint8_t* data;
    uint16_t size;
} hypervector_t;

typedef struct {
    float scores[16];  // Support up to 16 classes
    uint8_t num_classes;
} class_scores_t;

typedef struct {
    uint8_t* projection_matrix;
    uint16_t input_dim;
    uint16_t hv_dim;
    uint8_t quantization_bits;
} hdc_encoder_t;

typedef struct {
    uint8_t* class_prototypes;
    float* calibration_scores;
    uint16_t calibration_size;
    float alpha;
    uint8_t score_type;
} conformal_model_t;

// Core HDC operations (binary/ternary optimized for MCUs)
void hc_encode_binary(const uint8_t* input, hypervector_t* output, const hdc_encoder_t* encoder);
void hc_encode_ternary(const uint8_t* input, hypervector_t* output, const hdc_encoder_t* encoder);

// Fast distance computations using bit operations
uint16_t hamming_distance(const hypervector_t* hv1, const hypervector_t* hv2);
float cosine_similarity(const hypervector_t* hv1, const hypervector_t* hv2);

// Classification with conformal prediction (no floating point required)
void hc_classify(const hypervector_t* input_hv, const conformal_model_t* model, class_scores_t* scores);
void hc_conformal_predict(const class_scores_t* scores, const conformal_model_t* model, 
                         uint8_t* prediction, uint8_t* confidence);

// Memory management for constrained environments
int hc_init_model(conformal_model_t* model, const uint8_t* model_blob, size_t blob_size);
void hc_cleanup_model(conformal_model_t* model);
size_t hc_get_memory_usage(const conformal_model_t* model);

// Streaming calibration for online learning
void hc_update_calibration(conformal_model_t* model, const class_scores_t* new_scores, uint8_t true_label);
float hc_get_current_coverage(const conformal_model_t* model);

// Energy-aware prediction for battery-powered devices
typedef struct {
    uint32_t xor_ops;
    uint32_t popcount_ops;
    uint32_t memory_accesses;
    float estimated_energy_uj;
} energy_profile_t;

void hc_predict_with_energy(const hypervector_t* input_hv, const conformal_model_t* model,
                           class_scores_t* scores, energy_profile_t* energy);

#ifdef __cplusplus
}
#endif

#endif // HYPERCONFORMAL_H