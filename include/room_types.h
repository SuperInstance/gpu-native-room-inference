// room_types.h - Room data structures and types for GPU-native room inference

#ifndef ROOM_TYPES_H
#define ROOM_TYPES_H

#include <cuda_fp16.h>

// Room configuration
typedef struct {
    int input_dim;      // Input dimension per room
    int output_dim;     // Output dimension per room
    int context_size;   // Context size for room state
    int max_rooms;      // Maximum rooms supported
} RoomConfig;

// Room context (state maintained per room)
typedef struct {
    half* context;      // Room context buffer
    int context_len;    // Length of context
    int timestamp;      // Last update timestamp
    int room_id;        // Room identifier
} RoomContext;

// Warp room group (warp managing multiple rooms)
typedef struct {
    int warp_id;        // Warp identifier
    int room_ids[32];   // Room IDs managed by this warp (32 threads)
    int active_count;   // Number of active rooms in warp
} WarpRoomGroup;

// Performance metrics
typedef struct {
    float latency_ms;   // Inference latency in milliseconds
    float throughput_qps; // Queries per second
    float memory_mb;    // Memory usage in MB
    float power_w;      // Power consumption in watts
    float accuracy;     // Accuracy percentage
} RoomMetrics;

// Error codes
typedef enum {
    ROOM_SUCCESS = 0,
    ROOM_ERROR_INVALID_CONFIG,
    ROOM_ERROR_MEMORY_ALLOC,
    ROOM_ERROR_GPU_LAUNCH,
    ROOM_ERROR_SYNC_FAILED,
    ROOM_ERROR_CONTEXT_FULL
} RoomError;

// Room inference result
typedef struct {
    half* outputs;      // Inference outputs
    RoomMetrics metrics; // Performance metrics
    RoomError error;    // Error code
    int room_id;        // Room identifier
} RoomResult;

#endif // ROOM_TYPES_H
