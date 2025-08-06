/**
 * Arduino Nano 33 BLE Example - HyperConformal Classification
 * 
 * This example demonstrates ultra-low-power gesture recognition using
 * the onboard IMU sensor with HDC encoding and conformal prediction.
 * 
 * Hardware: Arduino Nano 33 BLE (ARM Cortex-M4F)
 * Memory: 256KB Flash, 64KB RAM
 * Power Target: <1mW average
 */

#ifdef ARDUINO
#include <Arduino.h>
#include <Arduino_LSM9DS1.h>
#include <EEPROM.h>
#endif

#include "hyperconformal.h"
#include <stdint.h>
#include <string.h>

// Pre-trained model for 5 gestures (up, down, left, right, circle)
// Compressed model blob (stored in program memory)
const uint8_t gesture_model_blob[] PROGMEM = {
    0x05,                    // num_classes = 5
    0x9A, 0x99, 0x99, 0x3E, // alpha = 0.15 (90% coverage)
    0x00,                    // score_type = APS
    0x64, 0x00,              // calibration_size = 100
    
    // Calibration scores (100 floats, simplified for demo)
    // ... (calibration data would be here in real deployment)
    
    // Class prototypes (5 classes × 1250 bytes each for 10K-dim binary HVs)
    // ... (prototype data would be here in real deployment)
};

// Global model and encoder instances
static conformal_model_t gesture_model;
static hdc_encoder_t imu_encoder;
static uint8_t projection_matrix[9 * 1250];  // 9 IMU features → 10K binary HV

// Circular buffer for IMU readings (gesture window)
#define GESTURE_WINDOW_SIZE 32
#define IMU_FEATURES 9  // 3x accel + 3x gyro + 3x mag

static float imu_buffer[GESTURE_WINDOW_SIZE][IMU_FEATURES];
static uint8_t buffer_index = 0;

// Energy monitoring
static uint32_t total_predictions = 0;
static float total_energy_uj = 0.0f;

// Initialize hardware and model
void setup() {
    Serial.begin(115200);
    while (!Serial);
    
    Serial.println("HyperConformal Gesture Recognition");
    Serial.println("===================================");
    
    // Initialize IMU
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }
    
    // Initialize encoder
    imu_encoder.input_dim = IMU_FEATURES;
    imu_encoder.hv_dim = 10000;
    imu_encoder.quantization_bits = 1;  // Binary
    imu_encoder.projection_matrix = projection_matrix;
    
    // Load projection matrix from EEPROM or generate random
    loadOrGenerateProjectionMatrix();
    
    // Initialize model from blob
    if (hc_init_model(&gesture_model, gesture_model_blob, sizeof(gesture_model_blob)) != 0) {
        Serial.println("Failed to initialize model!");
        while (1);
    }
    
    Serial.print("Model initialized. Memory usage: ");
    Serial.print(hc_get_memory_usage(&gesture_model));
    Serial.println(" bytes");
    
    Serial.println("Ready for gesture recognition!");
    Serial.println("Perform gestures to see real-time classification with confidence...");
}

// Main recognition loop
void loop() {
    // Read IMU data if available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
        float ax, ay, az, gx, gy, gz, mx, my, mz;
        
        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);
        IMU.readMagneticField(mx, my, mz);
        
        // Store in circular buffer
        imu_buffer[buffer_index][0] = ax;
        imu_buffer[buffer_index][1] = ay;
        imu_buffer[buffer_index][2] = az;
        imu_buffer[buffer_index][3] = gx;
        imu_buffer[buffer_index][4] = gy;
        imu_buffer[buffer_index][5] = gz;
        imu_buffer[buffer_index][6] = mx;
        imu_buffer[buffer_index][7] = my;
        imu_buffer[buffer_index][8] = mz;
        
        buffer_index = (buffer_index + 1) % GESTURE_WINDOW_SIZE;
        
        // Every 8 samples, perform gesture recognition
        if (buffer_index % 8 == 0) {
            performGestureRecognition();
        }
    }
    
    delay(20);  // ~50Hz sampling rate
}

// Perform gesture recognition with energy profiling
void performGestureRecognition() {
    // Extract features from IMU window (simple statistical features)
    uint8_t features[IMU_FEATURES];
    extractFeatures(features);
    
    // Encode to hypervector
    hypervector_t gesture_hv;
    hc_encode_binary(features, &gesture_hv, &imu_encoder);
    
    // Classify with energy profiling
    class_scores_t scores;
    energy_profile_t energy;
    hc_predict_with_energy(&gesture_hv, &gesture_model, &scores, &energy);
    
    // Get conformal prediction
    uint8_t predicted_gesture;
    uint8_t confidence;
    hc_conformal_predict(&scores, &gesture_model, &predicted_gesture, &confidence);
    
    // Update energy statistics
    total_predictions++;
    total_energy_uj += energy.estimated_energy_uj;
    
    // Display results only for high-confidence predictions
    if (confidence > 200) {  // ~78% confidence threshold
        Serial.print("Gesture: ");
        Serial.print(getGestureName(predicted_gesture));
        Serial.print(" (confidence: ");
        Serial.print((confidence * 100) / 255);
        Serial.print("%, energy: ");
        Serial.print(energy.estimated_energy_uj, 3);
        Serial.println(" μJ)");
        
        // Blink LED for visual feedback
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
    }
    
    // Print energy statistics every 100 predictions
    if (total_predictions % 100 == 0) {
        float avg_energy = total_energy_uj / total_predictions;
        Serial.print("Average energy per prediction: ");
        Serial.print(avg_energy, 3);
        Serial.println(" μJ");
    }
    
    // Cleanup
    if (gesture_hv.data) {
        free(gesture_hv.data);
    }
}

// Extract statistical features from IMU window
void extractFeatures(uint8_t* features) {
    for (int feature = 0; feature < IMU_FEATURES; feature++) {
        // Compute mean of current feature across window
        float mean = 0.0f;
        for (int i = 0; i < GESTURE_WINDOW_SIZE; i++) {
            mean += imu_buffer[i][feature];
        }
        mean /= GESTURE_WINDOW_SIZE;
        
        // Quantize to 8-bit (simple linear scaling)
        features[feature] = (uint8_t)((mean + 4.0f) * 32.0f);  // Assume ±4g range
    }
}

// Get human-readable gesture name
const char* getGestureName(uint8_t gesture_id) {
    switch (gesture_id) {
        case 0: return "UP";
        case 1: return "DOWN";
        case 2: return "LEFT";
        case 3: return "RIGHT";
        case 4: return "CIRCLE";
        default: return "UNKNOWN";
    }
}

// Load projection matrix from EEPROM or generate random
void loadOrGenerateProjectionMatrix() {
    const uint16_t EEPROM_MAGIC = 0xHC01;
    uint16_t stored_magic;
    
    // Check if valid projection matrix exists in EEPROM
    EEPROM.get(0, stored_magic);
    
    if (stored_magic == EEPROM_MAGIC) {
        Serial.println("Loading projection matrix from EEPROM...");
        EEPROM.get(2, projection_matrix);
    } else {
        Serial.println("Generating new random projection matrix...");
        
        // Generate random binary projection matrix
        randomSeed(analogRead(A0));  // Seed with noise
        for (int i = 0; i < sizeof(projection_matrix); i++) {
            projection_matrix[i] = random(0, 256);
        }
        
        // Store in EEPROM for future use
        EEPROM.put(0, EEPROM_MAGIC);
        EEPROM.put(2, projection_matrix);
        
        Serial.println("Projection matrix saved to EEPROM.");
    }
}

// Non-Arduino main function for testing
#ifndef ARDUINO
int main() {
    // Simulate Arduino setup/loop for testing
    setup();
    
    for (int i = 0; i < 1000; i++) {
        loop();
        usleep(20000);  // 20ms delay
    }
    
    // Cleanup
    hc_cleanup_model(&gesture_model);
    
    return 0;
}
#endif