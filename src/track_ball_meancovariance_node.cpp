#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <shield_planner_msgs/MeanCovariance.h>
#include <sl/Camera.hpp>
#include <std_msgs/Header.h>

namespace
{

constexpr double kGravity = 9.81;
constexpr int kStateDim = 6;
constexpr int kMeasurementDim = 3;

struct Detection
{
  cv::Rect box;
  float confidence = 0.0f;
  bool valid = false;
};

struct Measurement
{
  double t = 0.0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
};

struct Letterbox
{
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
};

class TensorRtLogger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      ROS_WARN_STREAM("TensorRT: " << msg);
    }
  }
};

size_t volume(const nvinfer1::Dims& dims)
{
  size_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    v *= static_cast<size_t>(dims.d[i] > 0 ? dims.d[i] : 1);
  }
  return v;
}

float sigmoid(float x)
{
  return 1.0f / (1.0f + std::exp(-x));
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
  const float inter = static_cast<float>((a & b).area());
  const float uni = static_cast<float>(a.area() + b.area()) - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::string dimsToString(const nvinfer1::Dims& dims)
{
  std::ostringstream out;
  out << '[';
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << dims.d[i];
  }
  out << ']';
  return out.str();
}

std::vector<unsigned char> readFile(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open TensorRT engine: " + path);
  }
  return std::vector<unsigned char>(
      std::istreambuf_iterator<char>(file),
      std::istreambuf_iterator<char>());
}

cv::Rect shrinkBox(const cv::Rect& box, const cv::Size& frame_size)
{
  const double frame_area = static_cast<double>(frame_size.width) * frame_size.height;
  const double area_ratio = static_cast<double>(box.width) * box.height / frame_area;
  const double normalized = (area_ratio - 0.000395) / (0.01 - 0.000395);
  const double r = 0.8 + (0.2 - 0.8) / (1.0 + std::exp(-100.0 * (normalized - 0.3)));

  const int dx = static_cast<int>(std::ceil((1.0 - std::sqrt(r)) * 0.5 * box.width));
  const int dy = static_cast<int>(std::ceil((1.0 - std::sqrt(r)) * 0.5 * box.height));

  cv::Rect shrunk(box.x + dx, box.y + dy, box.width - 2 * dx, box.height - 2 * dy);
  shrunk &= cv::Rect(0, 0, frame_size.width, frame_size.height);
  return shrunk;
}

cv::Rect rectToPixels(const cv::Rect2f& box, const cv::Size& frame_size)
{
  const int x1 = std::max(0, static_cast<int>(std::floor(box.x)));
  const int y1 = std::max(0, static_cast<int>(std::floor(box.y)));
  const int x2 = std::min(frame_size.width, static_cast<int>(std::ceil(box.x + box.width)));
  const int y2 = std::min(frame_size.height, static_cast<int>(std::ceil(box.y + box.height)));
  if (x2 <= x1 || y2 <= y1) {
    return cv::Rect();
  }
  return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

class YoloTensorRtDetector
{
public:
  YoloTensorRtDetector(const std::string& engine_path,
                       float confidence_threshold,
                       bool use_letterbox)
    : confidence_threshold_(confidence_threshold)
    , use_letterbox_(use_letterbox)
  {
    const std::vector<unsigned char> engine_data = readFile(engine_path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
      throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("Failed to create TensorRT execution context");
    }

#if NV_TENSORRT_MAJOR >= 10
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        input_name_ = name;
      } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        output_name_ = name;
      }
    }
    if (input_name_.empty() || output_name_.empty()) {
      throw std::runtime_error("Expected one input and one output tensor in TensorRT engine");
    }
    input_dims_ = engine_->getTensorShape(input_name_.c_str());
    output_dims_ = engine_->getTensorShape(output_name_.c_str());
#else
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      if (engine_->bindingIsInput(i)) {
        input_index_ = i;
      } else {
        output_index_ = i;
      }
    }
    if (input_index_ < 0 || output_index_ < 0) {
      throw std::runtime_error("Expected one input and one output binding in TensorRT engine");
    }
    input_dims_ = engine_->getBindingDimensions(input_index_);
    output_dims_ = engine_->getBindingDimensions(output_index_);
#endif

    ROS_INFO_STREAM("TensorRT input dims: " << dimsToString(input_dims_));
    ROS_INFO_STREAM("TensorRT output dims: " << dimsToString(output_dims_));

    if (input_dims_.nbDims == 4) {
      input_c_ = input_dims_.d[1];
      input_h_ = input_dims_.d[2];
      input_w_ = input_dims_.d[3];
    } else if (input_dims_.nbDims == 3) {
      input_c_ = input_dims_.d[0];
      input_h_ = input_dims_.d[1];
      input_w_ = input_dims_.d[2];
    }
    if (input_c_ != 3 || input_h_ <= 0 || input_w_ <= 0) {
      throw std::runtime_error("TensorRT engine must have static CHW image input");
    }

    input_count_ = volume(input_dims_);
    output_count_ = volume(output_dims_);
    host_input_.resize(input_count_);
    host_output_.resize(output_count_);

    cudaStreamCreate(&stream_);
#if NV_TENSORRT_MAJOR >= 10
    cudaMalloc(&input_device_, input_count_ * sizeof(float));
    cudaMalloc(&output_device_, output_count_ * sizeof(float));
#else
    cudaMalloc(&device_bindings_[input_index_], input_count_ * sizeof(float));
    cudaMalloc(&device_bindings_[output_index_], output_count_ * sizeof(float));
#endif
  }

  ~YoloTensorRtDetector()
  {
#if NV_TENSORRT_MAJOR >= 10
    if (input_device_) {
      cudaFree(input_device_);
    }
    if (output_device_) {
      cudaFree(output_device_);
    }
#else
    if (device_bindings_[input_index_]) {
      cudaFree(device_bindings_[input_index_]);
    }
    if (device_bindings_[output_index_]) {
      cudaFree(device_bindings_[output_index_]);
    }
#endif
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  Detection detect(const cv::Mat& rgb)
  {
    Letterbox lb;
    preprocess(rgb, lb);

#if NV_TENSORRT_MAJOR >= 10
    cudaMemcpyAsync(input_device_, host_input_.data(),
                    input_count_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->setTensorAddress(input_name_.c_str(), input_device_);
    context_->setTensorAddress(output_name_.c_str(), output_device_);
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(host_output_.data(), output_device_,
                    output_count_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
#else
    cudaMemcpyAsync(device_bindings_[input_index_], host_input_.data(),
                    input_count_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->enqueueV2(device_bindings_, stream_, nullptr);
    cudaMemcpyAsync(host_output_.data(), device_bindings_[output_index_],
                    output_count_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
#endif
    cudaStreamSynchronize(stream_);

    return decode(rgb.size(), lb);
  }

private:
  struct Candidate
  {
    cv::Rect2f box;
    float score = 0.0f;
  };

  void preprocess(const cv::Mat& rgb, Letterbox& lb)
  {
    cv::Mat canvas;
    if (use_letterbox_) {
      const float scale = std::min(static_cast<float>(input_w_) / rgb.cols,
                                   static_cast<float>(input_h_) / rgb.rows);
      const int resized_w = static_cast<int>(std::round(rgb.cols * scale));
      const int resized_h = static_cast<int>(std::round(rgb.rows * scale));
      lb.scale_x = scale;
      lb.scale_y = scale;
      lb.pad_x = (input_w_ - resized_w) / 2;
      lb.pad_y = (input_h_ - resized_h) / 2;

      cv::Mat resized;
      cv::resize(rgb, resized, cv::Size(resized_w, resized_h));
      canvas = cv::Mat(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
      resized.copyTo(canvas(cv::Rect(lb.pad_x, lb.pad_y, resized_w, resized_h)));
    } else {
      lb.scale_x = static_cast<float>(input_w_) / rgb.cols;
      lb.scale_y = static_cast<float>(input_h_) / rgb.rows;
      lb.pad_x = 0;
      lb.pad_y = 0;
      cv::resize(rgb, canvas, cv::Size(input_w_, input_h_));
    }

    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < input_h_; ++y) {
        for (int x = 0; x < input_w_; ++x) {
          host_input_[c * input_h_ * input_w_ + y * input_w_ + x] =
              canvas.at<cv::Vec3b>(y, x)[c] / 255.0f;
        }
      }
    }
  }

  Detection decode(const cv::Size& image_size, const Letterbox& lb) const
  {
    std::vector<Candidate> candidates;
    int attributes = 0;
    int boxes = 0;
    bool transposed = false;

    if (output_dims_.nbDims == 3) {
      const int a = output_dims_.d[1];
      const int b = output_dims_.d[2];
      if (a < b) {
        attributes = a;
        boxes = b;
        transposed = true;
      } else {
        boxes = a;
        attributes = b;
      }
    } else if (output_dims_.nbDims == 2) {
      const int a = output_dims_.d[0];
      const int b = output_dims_.d[1];
      if (a < b) {
        attributes = a;
        boxes = b;
        transposed = true;
      } else {
        boxes = a;
        attributes = b;
      }
    }

    if (attributes < 5 || boxes <= 0) {
      ROS_WARN_THROTTLE(1.0, "Unsupported YOLO output shape");
      return Detection{};
    }

    auto valueAt = [&](int box, int attr) -> float {
      if (transposed) {
        return host_output_[attr * boxes + box];
      }
      return host_output_[box * attributes + attr];
    };

    float best_score_seen = 0.0f;
    cv::Rect2f best_box_seen;
    for (int i = 0; i < boxes; ++i) {
      const float x = valueAt(i, 0);
      const float y = valueAt(i, 1);
      const float w = valueAt(i, 2);
      const float h = valueAt(i, 3);
      const float raw_score = attributes == 5 ? valueAt(i, 4) : valueAt(i, 4);
      const float score = raw_score > 1.0f || raw_score < 0.0f ? sigmoid(raw_score) : raw_score;
      const float x1 = (x - 0.5f * w - lb.pad_x) / lb.scale_x;
      const float y1 = (y - 0.5f * h - lb.pad_y) / lb.scale_y;
      const float x2 = (x + 0.5f * w - lb.pad_x) / lb.scale_x;
      const float y2 = (y + 0.5f * h - lb.pad_y) / lb.scale_y;
      cv::Rect2f box(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
      box &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(image_size.width),
                        static_cast<float>(image_size.height));
      if (score > best_score_seen) {
        best_score_seen = score;
        best_box_seen = box;
      }
      if (score < confidence_threshold_) {
        continue;
      }

      if (box.area() > 1.0f) {
        candidates.push_back({box, score});
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<Candidate> kept;
    for (const Candidate& candidate : candidates) {
      bool suppress = false;
      for (const Candidate& selected : kept) {
        if (iou(candidate.box, selected.box) > 0.45f) {
          suppress = true;
          break;
        }
      }
      if (!suppress) {
        kept.push_back(candidate);
      }
    }

    if (kept.empty()) {
      ROS_INFO_THROTTLE(1.0,
                        "YOLO decoded no boxes above threshold %.3f; best score seen %.3f best_box=[%.1f,%.1f %.1fx%.1f]",
                        confidence_threshold_, best_score_seen,
                        best_box_seen.x, best_box_seen.y,
                        best_box_seen.width, best_box_seen.height);
      return Detection{};
    }

    Detection detection;
    detection.box = rectToPixels(kept.front().box, image_size);
    detection.confidence = kept.front().score;
    detection.valid = detection.box.area() > 1;
    return detection;
  }

  struct RuntimeDeleter
  {
#if NV_TENSORRT_MAJOR >= 10
    void operator()(nvinfer1::IRuntime* p) const { delete p; }
#else
    void operator()(nvinfer1::IRuntime* p) const { if (p) p->destroy(); }
#endif
  };
  struct EngineDeleter
  {
#if NV_TENSORRT_MAJOR >= 10
    void operator()(nvinfer1::ICudaEngine* p) const { delete p; }
#else
    void operator()(nvinfer1::ICudaEngine* p) const { if (p) p->destroy(); }
#endif
  };
  struct ContextDeleter
  {
#if NV_TENSORRT_MAJOR >= 10
    void operator()(nvinfer1::IExecutionContext* p) const { delete p; }
#else
    void operator()(nvinfer1::IExecutionContext* p) const { if (p) p->destroy(); }
#endif
  };

  TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context_;
  std::string input_name_;
  std::string output_name_;
  nvinfer1::Dims input_dims_{};
  nvinfer1::Dims output_dims_{};
  int input_index_ = -1;
  int output_index_ = -1;
  int input_c_ = 0;
  int input_h_ = 0;
  int input_w_ = 0;
  size_t input_count_ = 0;
  size_t output_count_ = 0;
  std::vector<float> host_input_;
  std::vector<float> host_output_;
  void* input_device_ = nullptr;
  void* output_device_ = nullptr;
  void* device_bindings_[2] = {nullptr, nullptr};
  cudaStream_t stream_ = nullptr;
  float confidence_threshold_ = 0.3f;
  bool use_letterbox_ = true;
};

class SlidingKalman
{
public:
  SlidingKalman(int num_frame)
    : num_frame_(std::max(3, num_frame))
  {
  }

  void clear()
  {
    measurements_.clear();
  }

  void add(const Measurement& m)
  {
    measurements_.push_back(m);
    if (measurements_.size() > 10) {
      measurements_.erase(measurements_.begin());
    }
  }

  bool ready() const
  {
    return static_cast<int>(measurements_.size()) >= num_frame_;
  }

  bool estimate(Eigen::Matrix<double, kStateDim, 1>& mu,
                Eigen::Matrix<double, kStateDim, kStateDim>& covariance) const
  {
    if (measurements_.size() < 3) {
      return false;
    }

    const double dt0 = measurements_[2].t - measurements_[0].t;
    if (dt0 <= 1e-6) {
      return false;
    }

    Eigen::Matrix<double, kStateDim, 1> x;
    x.head<3>() = measurements_[2].position;
    x.tail<3>() = (measurements_[2].position - measurements_[0].position) / dt0;

    Eigen::Matrix<double, kStateDim, kStateDim> P =
        (Eigen::Matrix<double, kStateDim, 1>() << 1.0, 1.0, 1.0, 100.0, 100.0, 100.0).finished().asDiagonal();
    const Eigen::Matrix<double, kStateDim, kStateDim> Q =
        (Eigen::Matrix<double, kStateDim, 1>() << 0.01, 0.01, 0.01, 10.0, 10.0, 10.0).finished().asDiagonal();
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.01;

    Eigen::Matrix<double, kMeasurementDim, kStateDim> H;
    H.setZero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;

    for (size_t i = 2; i < measurements_.size(); ++i) {
      const double dt = measurements_[i].t - measurements_[i - 1].t;
      if (dt <= 1e-6) {
        continue;
      }

      Eigen::Matrix<double, kStateDim, kStateDim> F = Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
      F(0, 3) = dt;
      F(1, 4) = dt;
      F(2, 5) = dt;

      Eigen::Matrix<double, kStateDim, 1> u = Eigen::Matrix<double, kStateDim, 1>::Zero();
      u(2) = -kGravity * dt * dt;
      u(5) = -kGravity * dt;

      x = F * x + u;
      P = F * P * F.transpose() + Q;

      const Eigen::Vector3d residual = measurements_[i].position - H * x;
      const Eigen::Matrix3d S = H * P * H.transpose() + R;
      const Eigen::Matrix<double, kStateDim, kMeasurementDim> K = P * H.transpose() * S.inverse();
      x = x + K * residual;
      P = (Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H) * P;
    }

    mu = x;
    covariance = P;
    return true;
  }

  const std::vector<Measurement>& measurements() const
  {
    return measurements_;
  }

private:
  int num_frame_;
  std::vector<Measurement> measurements_;
};

std::vector<Eigen::Vector3d> filterPoints(const sl::Mat& point_cloud,
                                          const cv::Rect& box,
                                          double dist_threshold,
                                          const Eigen::Matrix4d& base_to_left)
{
  std::vector<Eigen::Vector3d> points;
  points.reserve(static_cast<size_t>(box.area()));

  const int x_end = box.x + box.width;
  const int y_end = box.y + box.height;
  for (int y = box.y; y < y_end; ++y) {
    for (int x = box.x; x < x_end; ++x) {
      sl::float4 point;
      point_cloud.getValue(x, y, &point);
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
        continue;
      }
      const Eigen::Vector4d left_point(point.x, point.y, point.z, 1.0);
      const Eigen::Vector3d p = (base_to_left * left_point).head<3>();
      if (p.x() > 0.8 && p.x() < dist_threshold &&
          p.y() > -2.0 && p.y() < 2.0 &&
          p.z() > 0.0 && p.z() < 2.8) {
        points.push_back(p);
      }
    }
  }

  if (points.empty()) {
    return points;
  }

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    mean += p;
  }
  mean /= static_cast<double>(points.size());

  Eigen::Vector3d variance = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    variance += (p - mean).array().square().matrix();
  }
  variance /= static_cast<double>(points.size());
  const Eigen::Vector3d stddev = variance.array().sqrt().matrix();

  std::vector<Eigen::Vector3d> filtered;
  filtered.reserve(points.size());
  for (const Eigen::Vector3d& p : points) {
    const bool keep =
        p.x() > mean.x() - 2.0 * stddev.x() && p.x() < mean.x() + 2.0 * stddev.x() &&
        p.y() > mean.y() - 2.0 * stddev.y() && p.y() < mean.y() + 2.0 * stddev.y() &&
        p.z() > mean.z() - 2.0 * stddev.z() && p.z() < mean.z() + 2.0 * stddev.z();
    if (keep) {
      filtered.push_back(p);
    }
  }
  return filtered;
}

Eigen::Vector3d meanPoint(const std::vector<Eigen::Vector3d>& points)
{
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    mean += p;
  }
  return mean / static_cast<double>(points.size());
}

}  // namespace

int main(int argc, char** argv)
{
  ros::init(argc, argv, "track_ball_meancovariance");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  double dist_threshold = 9.0;
  int min_pixel = 30;
  int num_frame = 4;
  std::string engine_path;
  std::string camera_resolution = "VGA";
  std::string svo_file;
  int camera_fps = 100;
  double confidence_threshold = 0.3;
  bool debug_log = false;
  bool use_letterbox = true;
  std::string debug_image_path;
  std::vector<double> base_to_left_values;

  pnh.param("dist_threshold", dist_threshold, dist_threshold);
  pnh.param("min_pixel", min_pixel, min_pixel);
  pnh.param("num_frame", num_frame, num_frame);
  pnh.param("engine_path", engine_path, engine_path);
  pnh.param("camera_resolution", camera_resolution, camera_resolution);
  pnh.param("camera_fps", camera_fps, camera_fps);
  pnh.param("svo_file", svo_file, svo_file);
  pnh.param("confidence_threshold", confidence_threshold, confidence_threshold);
  pnh.param("debug_log", debug_log, debug_log);
  pnh.param("use_letterbox", use_letterbox, use_letterbox);
  pnh.param("debug_image_path", debug_image_path, debug_image_path);

  if (!pnh.getParam("base_to_left", base_to_left_values)) {
    ROS_FATAL("~base_to_left is required; load it from params/zed2i.yaml");
    return 1;
  }
  if (base_to_left_values.size() != 16) {
    ROS_FATAL("~base_to_left must contain 16 row-major values, but contains %zu",
              base_to_left_values.size());
    return 1;
  }

  Eigen::Matrix4d base_to_left;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      const double value = base_to_left_values[static_cast<size_t>(row * 4 + col)];
      if (!std::isfinite(value)) {
        ROS_FATAL("~base_to_left contains a non-finite value at row %d, column %d", row, col);
        return 1;
      }
      base_to_left(row, col) = value;
    }
  }
  if (!base_to_left.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9)) {
    ROS_FATAL("~base_to_left must be a homogeneous transform with last row [0, 0, 0, 1]");
    return 1;
  }
  const Eigen::Matrix3d rotation = base_to_left.topLeftCorner<3, 3>();
  if (!(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-3) ||
      std::abs(rotation.determinant() - 1.0) > 1e-3) {
    ROS_FATAL("~base_to_left rotation block is not a valid rotation matrix");
    return 1;
  }

  if (num_frame < 3) {
    ROS_WARN("num_frame must be at least 3 for velocity initialization; using 3");
    num_frame = 3;
  }
  if (engine_path.empty()) {
    ROS_FATAL("~engine_path must point to a TensorRT YOLO engine file");
    return 1;
  }

  ROS_INFO("track_ball_meancovariance parameters: dist_threshold=%.3f, min_pixel=%d, num_frame=%d",
           dist_threshold, min_pixel, num_frame);

  ros::Publisher projectile_pub =
      nh.advertise<shield_planner_msgs::MeanCovariance>("projectile", 1);

  std::unique_ptr<YoloTensorRtDetector> detector;
  try {
    detector.reset(new YoloTensorRtDetector(engine_path,
                                            static_cast<float>(confidence_threshold),
                                            use_letterbox));
  } catch (const std::exception& e) {
    ROS_FATAL_STREAM(e.what());
    return 1;
  }

  sl::Camera zed;
  sl::InitParameters init;
  if (!svo_file.empty()) {
    init.input.setFromSVOFile(svo_file.c_str());
  }
  if (camera_resolution == "HD720") {
    init.camera_resolution = sl::RESOLUTION::HD720;
  } else {
    init.camera_resolution = sl::RESOLUTION::VGA;
  }
  init.camera_fps = camera_fps;
  init.depth_mode = sl::DEPTH_MODE::ULTRA;
  init.coordinate_units = sl::UNIT::METER;
  init.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;

  const sl::ERROR_CODE open_status = zed.open(init);
  if (open_status != sl::ERROR_CODE::SUCCESS) {
    ROS_FATAL_STREAM("Failed to open ZED camera: " << sl::toString(open_status));
    zed.close();
    return 1;
  }

  sl::RuntimeParameters runtime;
  runtime.confidence_threshold = 100;
  runtime.texture_confidence_threshold = 100;

  sl::Mat image;
  sl::Mat point_cloud;
  SlidingKalman kalman(num_frame);
  ros::Time first_stamp;
  int reset_count = 0;
  bool debug_image_written = false;

  while (ros::ok()) {
    if (zed.grab(runtime) != sl::ERROR_CODE::SUCCESS) {
      ros::spinOnce();
      continue;
    }

    const ros::Time stamp = ros::Time::now();
    zed.retrieveImage(image, sl::VIEW::LEFT);
    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

    cv::Mat rgba(static_cast<int>(image.getHeight()),
                 static_cast<int>(image.getWidth()),
                 CV_8UC4,
                 image.getPtr<sl::uchar1>(sl::MEM::CPU));
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_BGRA2RGB);
    if (!debug_image_written && !debug_image_path.empty()) {
      cv::Mat bgr;
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      if (cv::imwrite(debug_image_path, bgr)) {
        ROS_INFO_STREAM("Wrote debug RGB image to " << debug_image_path);
      } else {
        ROS_WARN_STREAM("Failed to write debug RGB image to " << debug_image_path);
      }
      debug_image_written = true;
    }

    const Detection detection = detector->detect(rgb);
    if (!detection.valid) {
      if (debug_log) {
        ROS_INFO_THROTTLE(1.0, "No YOLO ball detection");
      }
      if (++reset_count >= 5) {
        kalman.clear();
        first_stamp = ros::Time();
        reset_count = 0;
      }
      ros::spinOnce();
      continue;
    }

    const cv::Rect shrunk_box = shrinkBox(detection.box, rgb.size());
    const std::vector<Eigen::Vector3d> points =
        filterPoints(point_cloud, shrunk_box, dist_threshold, base_to_left);
    if (debug_log) {
      ROS_INFO_THROTTLE(1.0,
                        "Detection score=%.3f bbox=[%d,%d %dx%d] shrunk=[%d,%d %dx%d] filtered_points=%zu",
                        detection.confidence,
                        detection.box.x, detection.box.y, detection.box.width, detection.box.height,
                        shrunk_box.x, shrunk_box.y, shrunk_box.width, shrunk_box.height,
                        points.size());
    }
    if (static_cast<int>(points.size()) < min_pixel) {
      if (debug_log) {
        ROS_INFO_THROTTLE(1.0, "Skipping: filtered_points=%zu below min_pixel=%d",
                          points.size(), min_pixel);
      }
      if (++reset_count >= 5) {
        kalman.clear();
        first_stamp = ros::Time();
        reset_count = 0;
      }
      ros::spinOnce();
      continue;
    }
    reset_count = 0;

    if (first_stamp.isZero()) {
      first_stamp = stamp;
    }
    Measurement measurement;
    measurement.t = (stamp - first_stamp).toSec();
    measurement.position = meanPoint(points);
    kalman.add(measurement);
    if (debug_log) {
      ROS_INFO_THROTTLE(1.0, "Measurement t=%.4f pos=[%.3f %.3f %.3f] buffered=%zu/%d",
                        measurement.t,
                        measurement.position.x(), measurement.position.y(), measurement.position.z(),
                        kalman.measurements().size(), num_frame);
    }

    if (!kalman.ready()) {
      ros::spinOnce();
      continue;
    }

    Eigen::Matrix<double, kStateDim, 1> mu;
    Eigen::Matrix<double, kStateDim, kStateDim> covariance;
    if (!kalman.estimate(mu, covariance)) {
      ros::spinOnce();
      continue;
    }

    if (std::abs(mu(3)) <= 2.0) {
      if (debug_log) {
        ROS_INFO_THROTTLE(1.0, "Skipping publish: |vx|=%.3f <= 2.0, mu=[%.3f %.3f %.3f %.3f %.3f %.3f]",
                          std::abs(mu(3)), mu(0), mu(1), mu(2), mu(3), mu(4), mu(5));
      }
      ros::spinOnce();
      continue;
    }

    shield_planner_msgs::MeanCovariance msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "odom_combined";
    for (int i = 0; i < kStateDim; ++i) {
      msg.mu[i] = mu(i);
    }
    for (int r = 0; r < kStateDim; ++r) {
      for (int c = 0; c < kStateDim; ++c) {
        msg.P[r * kStateDim + c] = covariance(r, c);
      }
    }
    projectile_pub.publish(msg);
    ROS_WARN_THROTTLE(0.25, "Publishing projectile");

    ros::spinOnce();
  }

  zed.close();
  return 0;
}
