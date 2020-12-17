NGIF( size_t           , totalGlobalMem                        , " Global memory available on device in bytes " )
NGIF( size_t           , sharedMemPerBlock                     , " Shared memory available per block in bytes " )
NGIF( int              , regsPerBlock                          , " 32-bit registers available per block " )
NGIF( int              , warpSize                              , " Warp size in threads " )
NGIF( size_t           , memPitch                              , " Maximum pitch in bytes allowed by memory copies " )
NGIF( int              , maxThreadsPerBlock                    , " Maximum number of threads per block " )
NGIF( int              , clockRate                             , " Clock frequency in kilohertz " )
NGIF( size_t           , totalConstMem                         , " Constant memory available on device in bytes " )
NGIF( int              , major                                 , " Major compute capability " )
NGIF( int              , minor                                 , " Minor compute capability " )
NGIF( size_t           , textureAlignment                      , " Alignment requirement for textures " )
NGIF( size_t           , texturePitchAlignment                 , " Pitch alignment requirement for texture references bound to pitched memory " )
NGIF( int              , deviceOverlap                         , " Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. " )
NGIF( int              , multiProcessorCount                   , " Number of multiprocessors on device " )
NGIF( int              , kernelExecTimeoutEnabled              , " Specified whether there is a run time limit on kernels " )
NGIF( int              , integrated                            , " Device is integrated as opposed to discrete " )
NGIF( int              , canMapHostMemory                      , " Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer " )
NGIF( int              , computeMode                           , " Compute mode (See ::cudaComputeMode) " )
NGIF( int              , maxTexture1D                          , " Maximum 1D texture size " )
NGIF( int              , maxTexture1DMipmap                    , " Maximum 1D mipmapped texture size " )
NGIF( int              , maxTexture1DLinear                    , " Maximum size for 1D textures bound to linear memory " )
NGIF( A2               , maxTexture2D                          , " Maximum 2D texture dimensions " )
NGIF( A2               , maxTexture2DMipmap                    , " Maximum 2D mipmapped texture dimensions " )
NGIF( A3               , maxTexture2DLinear                    , " Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory " )
NGIF( A2               , maxTexture2DGather                    , " Maximum 2D texture dimensions if texture gather operations have to be performed " )
NGIF( A3               , maxTexture3D                          , " Maximum 3D texture dimensions " )
NGIF( A3               , maxTexture3DAlt                       , " Maximum alternate 3D texture dimensions " )
NGIF( int              , maxTextureCubemap                     , " Maximum Cubemap texture dimensions " )
NGIF( A2               , maxTexture1DLayered                   , " Maximum 1D layered texture dimensions " )
NGIF( A3               , maxTexture2DLayered                   , " Maximum 2D layered texture dimensions " )
NGIF( A2               , maxTextureCubemapLayered              , " Maximum Cubemap layered texture dimensions " )
NGIF( int              , maxSurface1D                          , " Maximum 1D surface size " )
NGIF( A2               , maxSurface2D                          , " Maximum 2D surface dimensions " )
NGIF( A3               , maxSurface3D                          , " Maximum 3D surface dimensions " )
NGIF( A2               , maxSurface1DLayered                   , " Maximum 1D layered surface dimensions " )
NGIF( A3               , maxSurface2DLayered                   , " Maximum 2D layered surface dimensions " )
NGIF( int              , maxSurfaceCubemap                     , " Maximum Cubemap surface dimensions " )
NGIF( A2               , maxSurfaceCubemapLayered              , " Maximum Cubemap layered surface dimensions " )
NGIF( size_t           , surfaceAlignment                      , " Alignment requirements for surfaces " )
NGIF( int              , concurrentKernels                     , " Device can possibly execute multiple kernels concurrently " )
NGIF( int              , ECCEnabled                            , " Device has ECC support enabled " )
NGIF( int              , pciBusID                              , " PCI bus ID of the device " )
NGIF( int              , pciDeviceID                           , " PCI device ID of the device " )
NGIF( int              , pciDomainID                           , " PCI domain ID of the device " )
NGIF( int              , tccDriver                             , " 1 if device is a Tesla device using TCC driver, 0 otherwise " )
NGIF( int              , asyncEngineCount                      , " Number of asynchronous engines " )
NGIF( int              , unifiedAddressing                     , " Device shares a unified address space with the host " )
NGIF( int              , memoryClockRate                       , " Peak memory clock frequency in kilohertz " )
NGIF( int              , memoryBusWidth                        , " Global memory bus width in bits " )
NGIF( int              , l2CacheSize                           , " Size of L2 cache in bytes " )
//NGIF( int            , persistingL2CacheMaxSize              , " Device's maximum l2 persisting lines capacity setting in bytes " )
NGIF( int              , maxThreadsPerMultiProcessor           , " Maximum resident threads per multiprocessor " )
NGIF( int              , streamPrioritiesSupported             , " Device supports stream priorities " )
NGIF( int              , globalL1CacheSupported                , " Device supports caching globals in L1 " )
NGIF( int              , localL1CacheSupported                 , " Device supports caching locals in L1 " )
NGIF( size_t           , sharedMemPerMultiprocessor            , " Shared memory available per multiprocessor in bytes " )
NGIF( int              , regsPerMultiprocessor                 , " 32-bit registers available per multiprocessor " )
NGIF( int              , managedMemory                         , " Device supports allocating managed memory on this system " )
NGIF( int              , isMultiGpuBoard                       , " Device is on a multi-GPU board " )
NGIF( int              , multiGpuBoardGroupID                  , " Unique identifier for a group of devices on the same multi-GPU board " )
NGIF( int              , hostNativeAtomicSupported             , " Link between the device and the host supports native atomic operations " )
NGIF( int              , singleToDoublePrecisionPerfRatio      , " Ratio of single precision performance (in floating-point operations per second) to double precision performance " )
NGIF( int              , pageableMemoryAccess                  , " Device supports coherently accessing pageable memory without calling cudaHostRegister on it " )
NGIF( int              , concurrentManagedAccess               , " Device can coherently access managed memory concurrently with the CPU " )
NGIF( int              , computePreemptionSupported            , " Device supports Compute Preemption " )
NGIF( int              , canUseHostPointerForRegisteredMem     , " Device can access host registered memory at the same virtual address as the CPU " )
NGIF( int              , cooperativeLaunch                     , " Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel " )
NGIF( int              , cooperativeMultiDeviceLaunch          , " Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice " )
NGIF( size_t           , sharedMemPerBlockOptin                , " Per device maximum shared memory per block usable by special opt in " )
NGIF( int              , pageableMemoryAccessUsesHostPageTables, " Device accesses pageable memory via the host's page tables " )
NGIF( int              , directManagedMemAccessFromHost        , " Host can directly access managed memory on the device without migration. " )
// NGIF( int           , maxBlocksPerMultiProcessor            , " Maximum number of resident blocks per multiprocessor " )
// NGIF( int           , accessPolicyMaxWindowSize             , " The maximum value of ::cudaAccessPolicyWindow::num_bytes. " )
// NGIF( size_t        , reservedSharedMemPerBlock             , " Shared memory reserved by CUDA driver per block in bytes " )