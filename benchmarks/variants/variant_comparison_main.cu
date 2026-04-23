// variant_comparison_main.cu - Compare all 8 warp variants

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "room_types.h"

// Forward declarations for each variant
extern "C" RoomError warp_lightweight_init(const RoomConfig* config, cudaStream_t stream);
extern "C" RoomError warp_high_throughput_init(const RoomConfig* config, cudaStream_t stream);
extern "C" RoomError warp_intelligent_init(const RoomConfig* config, cudaStream_t stream);
extern "C" RoomError warp_game_ai_init(const RoomConfig* config, cudaStream_t stream);
extern "C" RoomError warp_ultralight_init(const RoomConfig* config, cudaStream_t stream);
extern "C" RoomError warp_deterministic_init(const RoomConfig* config, cudaStream_t stream);
extern "C" RoomError warp_highprecision_init(const RoomConfig* config, cudaStream_t stream);
extern "C" RoomError warp_secure_init(const RoomConfig* config, cudaStream_t stream);

struct VariantInfo {
    std::string name;
    std::string domain;
    std::string description;
    RoomError (*init_func)(const RoomConfig*, cudaStream_t);
    std::vector<std::string> optimizations;
};

int main() {
    std::cout << "Warp Variant Comparison - All 8 Application Domains" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    // Common configuration
    RoomConfig config;
    config.input_dim = 256;
    config.output_dim = 128;
    config.context_size = 512;
    config.max_rooms = 1024;
    
    // All 8 variants
    std::vector<VariantInfo> variants = {
        {
            "warp_lightweight",
            "Edge AI",
            "Lightweight warp for edge devices (deckboss focus)",
            warp_lightweight_init,
            {"minimal_context", "energy_efficient", "fixed_assignments"}
        },
        {
            "warp_high_throughput", 
            "Cloud Serving",
            "High-throughput warp for cloud GPU instances",
            warp_high_throughput_init,
            {"persistent_kernels", "tensor_cores", "dynamic_batching", "multi_warp"}
        },
        {
            "warp_intelligent",
            "Scientific Simulation",
            "Intelligent warp with collective decision making",
            warp_intelligent_init,
            {"warp_voting", "collective_decisions", "adaptive_compute", "cross_room_attention"}
        },
        {
            "warp_game_ai",
            "Game AI",
            "Real-time warp for NPC coordination",
            warp_game_ai_init,
            {"real_time_scheduling", "behavior_coordination", "priority_queues", "group_tactics"}
        },
        {
            "warp_ultralight",
            "IoT & Sensors",
            "Ultra-lightweight warp for microcontrollers",
            warp_ultralight_init,
            {"minimal_memory", "energy_aware", "sleep_modes", "intermittent_connectivity"}
        },
        {
            "warp_deterministic",
            "Robotics",
            "Deterministic real-time warp for safety-critical systems",
            warp_deterministic_init,
            {"hard_real_time", "fault_tolerance", "safety_checks", "deterministic_execution"}
        },
        {
            "warp_highprecision",
            "Financial Modeling",
            "High-precision warp for numerical accuracy",
            warp_highprecision_init,
            {"double_precision", "audit_trails", "regulatory_checks", "monte_carlo_validation"}
        },
        {
            "warp_secure",
            "Healthcare & Medical AI",
            "Secure warp for privacy-preserving computation",
            warp_secure_init,
            {"differential_privacy", "access_control", "hipaa_compliance", "de_identification"}
        }
    };
    
    // Initialize and compare
    std::cout << "\nInitializing all 8 variants..." << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    std::cout << std::left << std::setw(20) << "Variant" 
              << std::setw(20) << "Domain" 
              << std::setw(10) << "Status" 
              << "Optimizations" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    for (const auto& variant : variants) {
        RoomError err = variant.init_func(&config, 0);
        
        std::cout << std::left << std::setw(20) << variant.name
                  << std::setw(20) << variant.domain
                  << std::setw(10) << (err == ROOM_SUCCESS ? "✅ OK" : "❌ FAIL");
        
        // List first 2 optimizations
        std::string optim_str;
        for (size_t i = 0; i < std::min(variant.optimizations.size(), size_t(2)); ++i) {
            if (i > 0) optim_str += ", ";
            optim_str += variant.optimizations[i];
        }
        if (variant.optimizations.size() > 2) {
            optim_str += ", ...";
        }
        
        std::cout << optim_str << std::endl;
    }
    
    // Summary
    std::cout << std::string(100, '=') << std::endl;
    std::cout << "\nSUMMARY: 8 Application Domain Variants" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    std::cout << "\nEdge Devices → Cloud → Scientific → Games → IoT → Robotics → Finance → Healthcare" << std::endl;
    std::cout << "\nArchitecture Coverage:" << std::endl;
    std::cout << "  • Performance: Edge AI (latency), Cloud (throughput)" << std::endl;
    std::cout << "  • Intelligence: Scientific (collective), Game AI (coordination)" << std::endl;
    std::cout << "  • Constraints: IoT (power), Robotics (real-time)" << std::endl;
    std::cout << "  • Compliance: Financial (accuracy), Healthcare (privacy)" << std::endl;
    
    std::cout << "\nCommon Warp-as-Room Principles:" << std::endl;
    std::cout << "  • GPU warp = PLATO room collective" << std::endl;
    std::cout << "  • Warp synchronization = room coordination" << std::endl;
    std::cout << "  • Domain-specific optimizations on common foundation" << std::endl;
    
    std::cout << "\nNext Steps:" << std::endl;
    std::cout << "  1. PLATO integration (bridge implemented)" << std::endl;
    std::cout << "  2. FM optimization challenge (cloud variant)" << std::endl;
    std::cout << "  3. Edge deployment validation (edge AI variant)" << std::endl;
    std::cout << "  4. Cross-domain benchmarking" << std::endl;
    
    std::cout << "\nVariant comparison completed successfully." << std::endl;
    return 0;
}
