// warp_api.h - Warp API for GPU-native room inference
// Standardized interface for warp-as-room architecture

#ifndef WARP_API_H
#define WARP_API_H

#include <cuda_runtime.h>
#include "room_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Warp Initialization and Configuration
// ============================================================================

/**
 * Initialize warp API with room configuration.
 * 
 * @param config Room configuration (input/output dimensions, context size)
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_init(const RoomConfig* config, cudaStream_t stream = 0);

/**
 * Configure warp scheduling policy.
 * 
 * @param policy Scheduling policy (0: round-robin, 1: priority, 2: affinity)
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_set_scheduling_policy(int policy, cudaStream_t stream = 0);

/**
 * Get current warp API configuration.
 * 
 * @param config Output: current configuration
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_get_config(RoomConfig* config);

// ============================================================================
// Room Scheduling and Management
// ============================================================================

/**
 * Schedule a room to a warp for processing.
 * 
 * @param room_id Room identifier
 * @param room_input Input data for the room [input_dim]
 * @param warp_id Output: warp assigned to this room
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_schedule_room(
    int room_id, const half* room_input, int* warp_id, cudaStream_t stream = 0);

/**
 * Schedule multiple rooms to warps (batch operation).
 * 
 * @param room_ids Array of room identifiers [num_rooms]
 * @param room_inputs Input data for all rooms [num_rooms x input_dim]
 * @param warp_ids Output: warp assignments for each room [num_rooms]
 * @param num_rooms Number of rooms to schedule
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_schedule_rooms_batch(
    const int* room_ids, const half* room_inputs, int* warp_ids,
    int num_rooms, cudaStream_t stream = 0);

/**
 * Get rooms currently assigned to a warp.
 * 
 * @param warp_id Warp identifier
 * @param room_ids Output: room IDs in this warp [32]
 * @param num_rooms Output: number of active rooms in warp
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_get_warp_rooms(int warp_id, int* room_ids, int* num_rooms);

// ============================================================================
// Warp Context and Synchronization
// ============================================================================

/**
 * Share room context within warp using warp shuffle.
 * 
 * @param warp_id Warp identifier
 * @param lane_id Lane within warp (0-31)
 * @param context Room context to share
 * @param context_size Size of context in elements
 * @param shared_context Output: shared context from all lanes
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_share_context(
    int warp_id, int lane_id, const half* context, int context_size,
    half* shared_context, cudaStream_t stream = 0);

/**
 * Perform warp-level synchronization.
 * 
 * @param warp_id Warp identifier
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_synchronize(int warp_id, cudaStream_t stream = 0);

/**
 * Warp collective operation: vote across warp.
 * 
 * @param warp_id Warp identifier
 * @param predicate Input predicate from each lane
 * @param result Output: warp vote result
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_vote(int warp_id, int predicate, int* result, cudaStream_t stream = 0);

// ============================================================================
// Room Inference Execution
// ============================================================================

/**
 * Execute room inference on assigned warp.
 * 
 * @param room_id Room identifier
 * @param weights Weight matrix [input_dim x output_dim]
 * @param room_output Output: inference results [output_dim]
 * @param metrics Output: performance metrics for this inference
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_execute_inference(
    int room_id, const half* weights, half* room_output,
    RoomMetrics* metrics, cudaStream_t stream = 0);

/**
 * Execute batch room inference.
 * 
 * @param room_ids Array of room identifiers [num_rooms]
 * @param weights Weight matrix [input_dim x output_dim]
 * @param room_outputs Output: inference results [num_rooms x output_dim]
 * @param metrics Output: performance metrics array [num_rooms]
 * @param num_rooms Number of rooms to infer
 * @param stream CUDA stream for asynchronous execution (optional)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_execute_inference_batch(
    const int* room_ids, const half* weights, half* room_outputs,
    RoomMetrics* metrics, int num_rooms, cudaStream_t stream = 0);

// ============================================================================
// Performance Monitoring and Metrics
// ============================================================================

/**
 * Get performance metrics for a warp.
 * 
 * @param warp_id Warp identifier
 * @param metrics Output: warp performance metrics
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_get_warp_metrics(int warp_id, RoomMetrics* metrics);

/**
 * Get performance metrics for all warps.
 * 
 * @param metrics Output: array of warp metrics [num_warps]
 * @param num_warps Input: size of metrics array
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_get_all_metrics(RoomMetrics* metrics, int num_warps);

/**
 * Reset performance metrics for a warp.
 * 
 * @param warp_id Warp identifier
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_reset_metrics(int warp_id);

// ============================================================================
// Error Handling and Fault Tolerance
// ============================================================================

/**
 * Get last error for a warp.
 * 
 * @param warp_id Warp identifier
 * @param error Output: last error code
 * @param error_msg Output: error message (if buffer provided)
 * @param error_msg_size Size of error message buffer
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_get_last_error(
    int warp_id, RoomError* error, char* error_msg, size_t error_msg_size);

/**
 * Recover warp from error state.
 * 
 * @param warp_id Warp identifier
 * @param recovery_policy Recovery policy (0: restart, 1: migrate, 2: ignore)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_recover_from_error(int warp_id, int recovery_policy);

/**
 * Migrate rooms from faulty warp to healthy warp.
 * 
 * @param src_warp_id Source warp (faulty)
 * @param dst_warp_id Destination warp (healthy)
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_migrate_rooms(int src_warp_id, int dst_warp_id);

// ============================================================================
// Cleanup and Resource Management
// ============================================================================

/**
 * Clean up warp API resources.
 * 
 * @return ROOM_SUCCESS on success, error code on failure
 */
RoomError warp_api_cleanup();

#ifdef __cplusplus
}
#endif

#endif // WARP_API_H
