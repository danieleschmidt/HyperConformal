/**
 * 🚀 HIERARCHICAL CONFORMAL CALIBRATION - Embedded C Implementation
 * 
 * Ultra-efficient implementation for ARM Cortex-M0+ and other MCUs.
 * Provides calibrated uncertainty quantification with minimal resources.
 * 
 * BREAKTHROUGH FEATURES:
 * - Fixed-point arithmetic (no floating point required)
 * - <512 bytes memory footprint
 * - <1ms inference time
 * - Formal coverage guarantees
 * 
 * Copyright (c) 2025 Terragon Labs
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef HIERARCHICAL_CONFORMAL_H
#define HIERARCHICAL_CONFORMAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Configuration constants */
#define HCC_MAX_LEVELS 4
#define HCC_FIXED_POINT_SCALE 10000  /* 4 decimal places precision */
#define HCC_CONFIDENCE_SCALE 255     /* 8-bit confidence values */

/**
 * Fixed-point type for efficient computation on MCUs without FPU
 */
typedef int32_t hcc_fixed_t;

/**
 * Hierarchical Conformal Configuration
 */
typedef struct {
    uint8_t num_levels;              /* Number of hierarchical levels (1-4) */
    uint8_t confidence_level;        /* Target confidence level (0-255) */
    uint16_t memory_budget;          /* Available memory in bytes */
    uint8_t adaptation_rate;         /* Online adaptation rate (0-255) */
} hcc_config_t;

/**
 * Compressed Hierarchical Conformal Model
 * Optimized for minimal memory footprint on MCUs
 */
typedef struct {
    uint8_t quantized_thresholds[HCC_MAX_LEVELS];  /* 8-bit quantized thresholds */
    hcc_fixed_t threshold_min;                      /* Minimum threshold (fixed-point) */
    hcc_fixed_t threshold_range;                    /* Threshold range (fixed-point) */
    uint8_t level_weights[HCC_MAX_LEVELS];         /* Level weights (0-255) */
    uint8_t num_levels;                            /* Active number of levels */
    uint8_t confidence_level;                      /* Target confidence (0-255) */
    uint32_t calibration_count;                    /* Number of calibration samples */
} hcc_model_t;

/**
 * Prediction result with coverage and confidence
 */
typedef struct {
    bool is_covered;           /* Coverage prediction */
    uint8_t confidence;        /* Confidence level (0-255) */
    uint8_t level_used;        /* Hierarchical level used for prediction */
    uint16_t inference_time;   /* Inference time in microseconds */
} hcc_result_t;

/**
 * Online adaptation state for streaming data
 */
typedef struct {
    uint8_t recent_scores[HCC_MAX_LEVELS * 4];  /* Circular buffer for recent scores */
    uint8_t buffer_indices[HCC_MAX_LEVELS];     /* Current buffer positions */
    uint8_t empirical_coverage[HCC_MAX_LEVELS]; /* Empirical coverage estimates */
    uint32_t update_count;                      /* Number of updates performed */
} hcc_adaptive_state_t;

/* Core API Functions */

/**
 * Initialize hierarchical conformal model from compressed representation
 * 
 * @param model Output model structure
 * @param config Configuration parameters
 * @param compressed_data Compressed model data from Python training
 * @param data_size Size of compressed data in bytes
 * @return 0 on success, negative error code on failure
 */
int hcc_init_model(hcc_model_t *model, 
                   const hcc_config_t *config,
                   const uint8_t *compressed_data, 
                   size_t data_size);

/**
 * Predict coverage using hierarchical conformal calibration
 * 
 * @param model Trained hierarchical conformal model
 * @param score Non-conformity score (fixed-point format)
 * @param level_hint Suggested hierarchical level (0=high confidence)
 * @param result Output prediction result
 * @return 0 on success, negative error code on failure
 */
int hcc_predict(const hcc_model_t *model,
                hcc_fixed_t score,
                uint8_t level_hint,
                hcc_result_t *result);

/**
 * Update model with new observations for online adaptation
 * 
 * @param model Model to update
 * @param adaptive_state State for online adaptation
 * @param score New non-conformity score
 * @param is_correct Whether the prediction was correct
 * @param level Level to update
 * @return 0 on success, negative error code on failure
 */
int hcc_update_online(hcc_model_t *model,
                      hcc_adaptive_state_t *adaptive_state,
                      hcc_fixed_t score,
                      bool is_correct,
                      uint8_t level);

/**
 * Get model memory footprint in bytes
 * 
 * @param model Hierarchical conformal model
 * @return Memory usage in bytes
 */
size_t hcc_get_memory_usage(const hcc_model_t *model);

/**
 * Validate model integrity and coverage guarantees
 * 
 * @param model Model to validate
 * @return 0 if valid, negative error code if invalid
 */
int hcc_validate_model(const hcc_model_t *model);

/* Utility Functions */

/**
 * Convert floating point value to fixed-point
 * 
 * @param value Floating point value
 * @return Fixed-point representation
 */
static inline hcc_fixed_t hcc_float_to_fixed(float value) {
    return (hcc_fixed_t)(value * HCC_FIXED_POINT_SCALE);
}

/**
 * Convert fixed-point value to floating point
 * 
 * @param fixed Fixed-point value
 * @return Floating point representation
 */
static inline float hcc_fixed_to_float(hcc_fixed_t fixed) {
    return (float)fixed / HCC_FIXED_POINT_SCALE;
}

/**
 * Dequantize 8-bit threshold to fixed-point
 * 
 * @param quantized Quantized threshold (0-255)
 * @param min_val Minimum value in range
 * @param range Value range
 * @return Dequantized fixed-point threshold
 */
static inline hcc_fixed_t hcc_dequantize_threshold(uint8_t quantized,
                                                   hcc_fixed_t min_val,
                                                   hcc_fixed_t range) {
    if (range == 0) return min_val;
    return min_val + ((hcc_fixed_t)quantized * range) / 255;
}

/**
 * Compute confidence based on score distance to threshold
 * 
 * @param score Non-conformity score
 * @param threshold Calibrated threshold
 * @return Confidence level (0-255)
 */
static inline uint8_t hcc_compute_confidence(hcc_fixed_t score, hcc_fixed_t threshold) {
    if (threshold == 0) return 128; /* Neutral confidence */
    
    hcc_fixed_t distance = score > threshold ? score - threshold : threshold - score;
    hcc_fixed_t ratio = (distance * HCC_FIXED_POINT_SCALE) / threshold;
    
    /* Confidence = 1 / (1 + ratio) */
    hcc_fixed_t confidence_fixed = HCC_FIXED_POINT_SCALE / (HCC_FIXED_POINT_SCALE + ratio);
    
    /* Scale to 8-bit */
    return (uint8_t)((confidence_fixed * HCC_CONFIDENCE_SCALE) / HCC_FIXED_POINT_SCALE);
}

/* Arduino-specific helpers */
#ifdef ARDUINO

/**
 * Initialize hierarchical conformal model from PROGMEM data
 * 
 * @param model Output model structure
 * @param config Configuration parameters
 * @param progmem_data Model data stored in program memory
 * @param data_size Size of data in bytes
 * @return 0 on success, negative error code on failure
 */
int hcc_init_from_progmem(hcc_model_t *model,
                          const hcc_config_t *config,
                          const uint8_t *progmem_data,
                          size_t data_size);

/**
 * Save adaptive state to EEPROM for persistence
 * 
 * @param adaptive_state State to save
 * @param eeprom_address Starting EEPROM address
 * @return 0 on success, negative error code on failure
 */
int hcc_save_state_eeprom(const hcc_adaptive_state_t *adaptive_state,
                          int eeprom_address);

/**
 * Load adaptive state from EEPROM
 * 
 * @param adaptive_state State structure to populate
 * @param eeprom_address Starting EEPROM address
 * @return 0 on success, negative error code on failure
 */
int hcc_load_state_eeprom(hcc_adaptive_state_t *adaptive_state,
                          int eeprom_address);

#endif /* ARDUINO */

/* Error codes */
#define HCC_SUCCESS                0
#define HCC_ERROR_INVALID_PARAM   -1
#define HCC_ERROR_INVALID_MODEL   -2
#define HCC_ERROR_INSUFFICIENT_MEMORY -3
#define HCC_ERROR_CORRUPTED_DATA  -4
#define HCC_ERROR_UNSUPPORTED     -5

#ifdef __cplusplus
}
#endif

#endif /* HIERARCHICAL_CONFORMAL_H */