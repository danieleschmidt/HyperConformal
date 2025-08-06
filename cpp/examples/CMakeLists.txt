# Examples for HyperConformal C library

# Arduino example
add_executable(arduino_example arduino_nano_33_ble.c)
target_link_libraries(arduino_example hyperconformal)

# STM32 example  
add_executable(stm32_example stm32f4_discovery.c)
target_link_libraries(stm32_example hyperconformal)

# Raspberry Pi Pico example
add_executable(pico_example raspberry_pi_pico.c)
target_link_libraries(pico_example hyperconformal)

# Desktop benchmark
if(NOT CMAKE_CROSSCOMPILING)
    add_executable(benchmark benchmark.c)
    target_link_libraries(benchmark hyperconformal m)
    
    add_executable(energy_profiler energy_profiler.c)
    target_link_libraries(energy_profiler hyperconformal m)
endif()