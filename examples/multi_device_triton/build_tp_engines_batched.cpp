// Batched TP weight-shard + engine builder (TRT-28040 follow-on; Morpheus-shaped).
// Same Megatron MLP as build_tp_engines.cpp but with a DYNAMIC batch dimension so the
// model is a proper Triton/Morpheus batched model (max_batch_size > 0):
//   X[-1,K] @ W1[K,Fr] -> H[-1,Fr] @ W2[Fr,N] -> Y[-1,N], AllReduce(SUM) across ranks.
// Triton config: max_batch_size=MAXB, input "X" dims [K], output "Y" dims [N].
//
//   nvcc -std=c++17 -ccbin g++ -w build_tp_engines_batched.cpp -o build_tp_engines_batched \
//        -I$TRT/include -L$TRT/lib -lnvinfer
//   LD_LIBRARY_PATH=$TRT/lib ./build_tp_engines_batched <world> <out_dir>
#include <NvInfer.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity s, const char* m) noexcept override
  { if (s <= Severity::kERROR) fprintf(stderr, "[TRT] %s\n", m); }
} gLogger;

constexpr int MAXB = 2048, K = 4096, F = 8192, N = 4096;  // max batch, hidden, inter, out
static std::vector<float> W1, W2;

static void initWeights() {
  W1.resize((size_t)K * F); W2.resize((size_t)F * N);
  for (size_t i = 0; i < W1.size(); ++i) W1[i] = 0.01f * ((int)(i % 17) - 8) / std::sqrt((float)K);
  for (size_t i = 0; i < W2.size(); ++i) W2[i] = 0.01f * ((int)(i % 13) - 6) / std::sqrt((float)F);
}

static std::vector<char> buildEngine(int world, int rank) {
  const int Fr = (world == 1) ? F : F / world;
  const int f0 = (world == 1) ? 0 : rank * Fr;
  std::vector<float> w1(K * Fr), w2(Fr * N);
  for (int k = 0; k < K; ++k)
    for (int f = 0; f < Fr; ++f) w1[k * Fr + f] = W1[(size_t)k * F + (f0 + f)];
  for (int f = 0; f < Fr; ++f)
    for (int n = 0; n < N; ++n) w2[(size_t)f * N + n] = W2[(size_t)(f0 + f) * N + n];

  std::unique_ptr<IBuilder> builder(createInferBuilder(gLogger));
  std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(
      1U << (uint32_t)NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
  ITensor* x = net->addInput("X", DataType::kFLOAT, Dims2{-1, K});   // dynamic batch
  auto* c1 = net->addConstant(Dims2{K, Fr}, Weights{DataType::kFLOAT, w1.data(), (int64_t)w1.size()});
  auto* h = net->addMatrixMultiply(*x, MatrixOperation::kNONE, *c1->getOutput(0), MatrixOperation::kNONE);
  auto* c2 = net->addConstant(Dims2{Fr, N}, Weights{DataType::kFLOAT, w2.data(), (int64_t)w2.size()});
  auto* p = net->addMatrixMultiply(*h->getOutput(0), MatrixOperation::kNONE, *c2->getOutput(0), MatrixOperation::kNONE);
  ITensor* y = p->getOutput(0);
  if (world > 1) {
    auto* coll = net->addDistCollective(*y, CollectiveOperation::kALL_REDUCE, ReduceOperation::kSUM, -1, nullptr, 0);
    coll->setNbRanks(world);
    y = coll->getOutput(0);
  }
  y->setName("Y");
  net->markOutput(*y);

  std::unique_ptr<IBuilderConfig> cfg(builder->createBuilderConfig());
  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("X", OptProfileSelector::kMIN, Dims2{1, K});
  profile->setDimensions("X", OptProfileSelector::kOPT, Dims2{MAXB, K});
  profile->setDimensions("X", OptProfileSelector::kMAX, Dims2{MAXB, K});
  cfg->addOptimizationProfile(profile);

  std::unique_ptr<IHostMemory> ser(builder->buildSerializedNetwork(*net, *cfg));
  if (!ser) { fprintf(stderr, "build failed world=%d rank=%d\n", world, rank); std::abort(); }
  const char* d = static_cast<const char*>(ser->data());
  return std::vector<char>(d, d + ser->size());
}

int main(int argc, char** argv) {
  int world = (argc > 1) ? atoi(argv[1]) : 2;
  std::string dir = (argc > 2) ? argv[2] : ".";
  initWeights();
  { auto e = buildEngine(1, 0);
    std::ofstream f(dir + "/model_sd.plan", std::ios::binary); f.write(e.data(), e.size());
    printf("wrote %s/model_sd.plan (%zu bytes, full weights, batched)\n", dir.c_str(), e.size()); }
  for (int r = 0; r < world; ++r) {
    auto e = buildEngine(world, r);
    std::ofstream f(dir + "/model.plan.rank" + std::to_string(r), std::ios::binary);
    f.write(e.data(), e.size());
    printf("wrote %s/model.plan.rank%d (%zu bytes, weight shard, batched)\n", dir.c_str(), r, e.size());
  }
  printf("dims: X[-1,%d] -> Y[-1,%d], maxB=%d, F=%d across %d ranks (Fr=%d)\n", K, N, MAXB, F, world, F/world);
  return 0;
}
