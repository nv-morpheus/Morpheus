void doca_receive_persistent(
  doca_gpu_rxq_info*      rxq_info,
  doca_gpu_semaphore_in*  sem_in,
  uint32_t                sem_count,
  cudaStream_t stream
);
