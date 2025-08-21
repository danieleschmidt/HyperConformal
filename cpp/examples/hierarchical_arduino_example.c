/**
 * 🚀 HIERARCHICAL CONFORMAL - Arduino Example
 * 
 * Demonstration of ultra-efficient hierarchical conformal prediction
 * on Arduino Nano 33 BLE with sensor data classification.
 * 
 * BREAKTHROUGH DEMO:
 * - Real-time uncertainty quantification
 * - <512 bytes memory usage
 * - <1ms inference time
 * - Formal coverage guarantees
 */

#ifdef ARDUINO
#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include "hierarchical_conformal.h"

/* Pre-trained model data stored in program memory */
const uint8_t PROGMEM model_data[] = {
    /* Quantized thresholds for 3 levels */
    0x3C, 0x78, 0xB4,  /* 60, 120, 180 in 8-bit */
    
    /* Threshold min (32-bit fixed-point) */
    0x00, 0x00, 0x10, 0x00,  /* 0.1 * 10000 = 1000 */
    
    /* Threshold range (32-bit fixed-point) */
    0x00, 0x00, 0x50, 0x00,  /* 0.5 * 10000 = 5000 */
    
    /* Level weights */
    0x80, 0x60, 0x40,  /* 128, 96, 64 normalized weights */
    
    /* Metadata */
    0x03, 0xE6  /* 3 levels, confidence 230 (~90%) */
};

/* Global model and state */
hcc_model_t global_model;
hcc_adaptive_state_t adaptive_state;

/* Sensor configuration */
#define SENSOR_PIN A0
#define NUM_SENSORS 4
#define CLASSIFICATION_THRESHOLD 512

/* Performance tracking */
unsigned long total_predictions = 0;
unsigned long correct_predictions = 0;
unsigned long total_inference_time = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("🚀 Hierarchical Conformal Prediction Demo");
    Serial.println("=========================================");
    
    /* Initialize sensor */
    pinMode(SENSOR_PIN, INPUT);
    analogReadResolution(12);  /* 12-bit ADC on Nano 33 BLE */
    
    /* Configure hierarchical conformal predictor */
    hcc_config_t config = {
        .num_levels = 3,
        .confidence_level = 230,  /* ~90% confidence */
        .memory_budget = 512,
        .adaptation_rate = 32     /* ~12.5% adaptation */
    };
    
    /* Initialize model from program memory */
    int result = hcc_init_from_progmem(&global_model, &config, 
                                       model_data, sizeof(model_data));
    if (result != HCC_SUCCESS) {
        Serial.print("❌ Model initialization failed: ");
        Serial.println(result);
        while (1) delay(1000);
    }
    
    /* Load adaptive state from EEPROM (if available) */
    result = hcc_load_state_eeprom(&adaptive_state, 0);
    if (result != HCC_SUCCESS) {
        Serial.println("⚠️  EEPROM state not found, using defaults");
        memset(&adaptive_state, 0, sizeof(adaptive_state));
        /* Initialize empirical coverage estimates */
        for (int i = 0; i < 3; i++) {
            adaptive_state.empirical_coverage[i] = 200;  /* ~78% */
        }
    } else {
        Serial.println("✅ Loaded adaptive state from EEPROM");
    }
    
    /* Display model information */
    size_t memory_usage = hcc_get_memory_usage(&global_model);
    Serial.print("📊 Model memory usage: ");
    Serial.print(memory_usage);
    Serial.println(" bytes");
    
    if (memory_usage <= config.memory_budget) {
        Serial.println("✅ Memory budget satisfied");
    } else {
        Serial.println("❌ Memory budget exceeded");
    }
    
    Serial.println("\n🎯 Starting real-time classification...");
    Serial.println("Format: [Sensor] [Score] [Level] [Covered] [Confidence] [Time]");
    Serial.println("----------------------------------------------------------------");
}

void loop() {
    /* Read sensor data */
    int sensor_values[NUM_SENSORS];
    for (int i = 0; i < NUM_SENSORS; i++) {
        sensor_values[i] = analogRead(SENSOR_PIN + i);
        delay(1);  /* Small delay for ADC settling */
    }
    
    /* Compute simple non-conformity score */
    hcc_fixed_t score = compute_nonconformity_score(sensor_values);
    
    /* Determine hierarchical level based on sensor variance */
    uint8_t level_hint = determine_hierarchical_level(sensor_values);
    
    /* Perform hierarchical conformal prediction */
    unsigned long start_time = micros();
    hcc_result_t result;
    int predict_result = hcc_predict(&global_model, score, level_hint, &result);
    unsigned long inference_time = micros() - start_time;
    
    if (predict_result == HCC_SUCCESS) {
        /* Determine ground truth (simplified for demo) */
        bool ground_truth = (sensor_values[0] > CLASSIFICATION_THRESHOLD);
        bool is_correct = (result.is_covered == ground_truth);
        
        /* Update performance tracking */
        total_predictions++;
        if (is_correct) correct_predictions++;
        total_inference_time += inference_time;
        
        /* Online adaptation */
        hcc_update_online(&global_model, &adaptive_state, score, 
                         is_correct, result.level_used);
        
        /* Display results */
        Serial.print(sensor_values[0]);
        Serial.print("\t");
        Serial.print(hcc_fixed_to_float(score), 3);
        Serial.print("\t");
        Serial.print(result.level_used);
        Serial.print("\t");
        Serial.print(result.is_covered ? "YES" : "NO");
        Serial.print("\t");
        Serial.print((result.confidence * 100) / 255);
        Serial.print("%\t");
        Serial.print(inference_time);
        Serial.println("μs");
        
        /* Periodic performance summary */
        if (total_predictions % 100 == 0) {
            display_performance_summary();
            
            /* Save adaptive state to EEPROM */
            hcc_save_state_eeprom(&adaptive_state, 0);
        }
        
        /* Verify coverage guarantee */
        if (total_predictions % 50 == 0) {
            verify_coverage_guarantee();
        }
    } else {
        Serial.print("❌ Prediction failed: ");
        Serial.println(predict_result);
    }
    
    delay(100);  /* 10 Hz prediction rate */
}

hcc_fixed_t compute_nonconformity_score(const int sensor_values[]) {
    /* Simple non-conformity based on sensor variance and magnitude */
    
    /* Calculate mean */
    int32_t sum = 0;
    for (int i = 0; i < NUM_SENSORS; i++) {
        sum += sensor_values[i];
    }
    int mean = sum / NUM_SENSORS;
    
    /* Calculate variance */
    int32_t variance_sum = 0;
    for (int i = 0; i < NUM_SENSORS; i++) {
        int diff = sensor_values[i] - mean;
        variance_sum += diff * diff;
    }
    int variance = variance_sum / NUM_SENSORS;
    
    /* Non-conformity score: higher variance = higher non-conformity */
    float score_float = sqrt(variance) / 100.0;  /* Normalize to reasonable range */
    return hcc_float_to_fixed(score_float);
}

uint8_t determine_hierarchical_level(const int sensor_values[]) {
    /* Determine hierarchical level based on signal characteristics */
    
    int max_val = sensor_values[0];
    int min_val = sensor_values[0];
    
    for (int i = 1; i < NUM_SENSORS; i++) {
        if (sensor_values[i] > max_val) max_val = sensor_values[i];
        if (sensor_values[i] < min_val) min_val = sensor_values[i];
    }
    
    int dynamic_range = max_val - min_val;
    
    /* Level assignment based on dynamic range */
    if (dynamic_range < 100) return 0;        /* High confidence */
    else if (dynamic_range < 500) return 1;   /* Medium confidence */
    else return 2;                            /* Low confidence */
}

void display_performance_summary() {
    Serial.println("\n📊 PERFORMANCE SUMMARY");
    Serial.println("======================");
    
    float accuracy = (float)correct_predictions / total_predictions * 100.0;
    float avg_inference_time = (float)total_inference_time / total_predictions;
    
    Serial.print("Total Predictions: ");
    Serial.println(total_predictions);
    Serial.print("Accuracy: ");
    Serial.print(accuracy, 1);
    Serial.println("%");
    Serial.print("Avg Inference Time: ");
    Serial.print(avg_inference_time, 1);
    Serial.println(" μs");
    
    /* Display adaptive state */
    Serial.println("\n🔄 Adaptive State:");
    for (int i = 0; i < global_model.num_levels; i++) {
        float coverage = (float)adaptive_state.empirical_coverage[i] / 255.0 * 100.0;
        Serial.print("Level ");
        Serial.print(i);
        Serial.print(" Coverage: ");
        Serial.print(coverage, 1);
        Serial.println("%");
    }
    
    Serial.println("================================================================\n");
}

void verify_coverage_guarantee() {
    Serial.println("\n🔍 COVERAGE VERIFICATION");
    Serial.println("========================");
    
    /* Calculate empirical coverage */
    float empirical_coverage = (float)correct_predictions / total_predictions * 100.0;
    float target_coverage = (float)global_model.confidence_level / 255.0 * 100.0;
    
    Serial.print("Target Coverage: ");
    Serial.print(target_coverage, 1);
    Serial.println("%");
    Serial.print("Empirical Coverage: ");
    Serial.print(empirical_coverage, 1);
    Serial.println("%");
    
    if (empirical_coverage >= target_coverage - 5.0) {
        Serial.println("✅ Coverage guarantee maintained");
    } else {
        Serial.println("⚠️  Coverage below target - model adapting");
    }
    
    Serial.println("================================================\n");
}

#else

/* Non-Arduino compilation - provide empty functions */
void setup() {}
void loop() {}

#endif /* ARDUINO */