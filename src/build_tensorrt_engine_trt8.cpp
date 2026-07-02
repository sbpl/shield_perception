#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace
{

class Logger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kINFO) {
      std::cerr << "[TRT] " << msg << '\n';
    }
  }
};

template <typename T>
struct TrtDestroy
{
  void operator()(T* ptr) const
  {
    if (ptr != nullptr) {
      ptr->destroy();
    }
  }
};

std::vector<char> readFile(const std::string& path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open input file: " + path);
  }
  return std::vector<char>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, const void* data, std::size_t size)
{
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Unable to open output file: " + path);
  }
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!output) {
    throw std::runtime_error("Failed while writing output file: " + path);
  }
}

void printUsage(const char* program)
{
  std::cerr << "Usage: " << program
            << " <model.onnx> <output.engine> [workspace_mib] [--fp16]\n";
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3 || argc > 5) {
    printUsage(argv[0]);
    return 1;
  }

  const std::string onnx_path = argv[1];
  const std::string engine_path = argv[2];
  int workspace_mib = 4096;
  bool use_fp16 = false;

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--fp16") {
      use_fp16 = true;
    } else {
      workspace_mib = std::atoi(arg.c_str());
    }
  }

  try {
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder, TrtDestroy<nvinfer1::IBuilder>> builder(
        nvinfer1::createInferBuilder(logger));
    if (!builder) {
      throw std::runtime_error("Failed to create TensorRT builder");
    }

    const uint32_t explicit_batch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroy<nvinfer1::INetworkDefinition>> network(
        builder->createNetworkV2(explicit_batch));
    if (!network) {
      throw std::runtime_error("Failed to create TensorRT network");
    }

    std::unique_ptr<nvonnxparser::IParser, TrtDestroy<nvonnxparser::IParser>> parser(
        nvonnxparser::createParser(*network, logger));
    if (!parser) {
      throw std::runtime_error("Failed to create ONNX parser");
    }

    const std::vector<char> onnx = readFile(onnx_path);
    if (!parser->parse(onnx.data(), onnx.size())) {
      std::cerr << "Failed to parse ONNX model:\n";
      for (int i = 0; i < parser->getNbErrors(); ++i) {
        std::cerr << parser->getError(i)->desc() << '\n';
      }
      return 1;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroy<nvinfer1::IBuilderConfig>> config(
        builder->createBuilderConfig());
    if (!config) {
      throw std::runtime_error("Failed to create TensorRT builder config");
    }
    config->setMaxWorkspaceSize(static_cast<std::size_t>(workspace_mib) << 20);

    if (use_fp16) {
      if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
      } else {
        std::cerr << "FP16 requested, but platform does not report fast FP16 support.\n";
      }
    }

    std::unique_ptr<nvinfer1::IHostMemory, TrtDestroy<nvinfer1::IHostMemory>> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      throw std::runtime_error("TensorRT engine build failed");
    }

    writeFile(engine_path, serialized->data(), serialized->size());
    std::cout << engine_path << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 0;
}
