/**************************************************************************************************

        2017/07/01 Sakuma Hiroki
        Structure From Motion

**************************************************************************************************/

#pragma once

#include <boost/filesystem.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/range.hpp>
#include <boost/signals2.hpp>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/*************************************************************************************************/

using index_type = int;

template <class Value>
using matrix = boost::multi_array<Value, 2>;

template <class Key, class Value, class Hash = std::hash<Key>,
          class Pred = std::equal_to<Key>>
using hash_map = std::unordered_map<Key, Value, Hash, Pred>;

template <class Key, class Hash = std::hash<Key>,
          class Pred = std::equal_to<Key>>
using hash_set = std::unordered_set<Key, Hash, Pred>;

namespace cv {
using Matx15d = cv::Matx<double, 1, 5>;
}

/*************************************************************************************************/

template <class Key>
struct Hash {
    size_t operator()(const Key& key) const {
        return std::hash<typename std::tuple_element<0, Key>::type>()(
                   key.first) ^
               std::hash<typename std::tuple_element<1, Key>::type>()(
                   key.second);
    }
};

/*************************************************************************************************/

class StructureFromMotion {
   public:
    StructureFromMotion() = default;
    virtual ~StructureFromMotion() = default;

    StructureFromMotion(const StructureFromMotion&) = delete;
    StructureFromMotion(StructureFromMotion&&) = delete;

    StructureFromMotion& operator=(const StructureFromMotion&) = delete;
    StructureFromMotion& operator=(StructureFromMotion&&) = delete;

    using Ptr = std::shared_ptr<StructureFromMotion>;

   private:
    cv::Matx33d IntrinsicMatrix;

    cv::Matx15d distortionCoefficients;

    std::vector<cv::Matx34d> ExtrinsicMatrixList;

    std::vector<cv::Mat> imageList;

    std::vector<std::vector<cv::KeyPoint>> keyPointsList;

    std::vector<std::vector<cv::Point2f>> pointsList;

    std::vector<cv::Mat> descriptorsList;

    matrix<std::vector<cv::DMatch>> matchesMatrix;

    std::vector<std::pair<cv::Point3f, cv::Vec3b>> pointCloud;

    std::vector<hash_map<index_type, index_type>> matches3d2d;

    hash_set<index_type> viewsNotProcessed;

    hash_set<index_type> viewsProcessed;

    boost::signals2::signal<void(
        const std::vector<std::pair<cv::Point3f, cv::Vec3b>>&)>
        signal;

    void initialize();

    void detectFeatures();

    void matchFeatures();

    void reconstructFoundation();

    void reconstructMoreViews();

    void findExtrinsicMatrices(index_type, index_type, std::vector<cv::DMatch>&,
                               std::vector<cv::DMatch>&, cv::Matx34d&,
                               cv::Matx34d&);

    void triangulateViews(index_type, index_type);

   public:
    static Ptr create() { return std::make_shared<StructureFromMotion>(); }

    boost::signals2::connection registerCallback(
        const boost::function<void(
            const std::vector<std::pair<cv::Point3f, cv::Vec3b>>&)>& callback) {
        return signal.connect(callback);
    }

    void loadImage(const boost::filesystem::path&);
    void clearImages() { imageList.clear(); }

    void loadParameter(const boost::filesystem::path&);

    void process();

    void setIntrinsicMatrix(const cv::Matx33d& _IntrinsicMatrix) {
        IntrinsicMatrix = _IntrinsicMatrix;
    }
    cv::Matx33d getIntrinsicMatrix() const { return IntrinsicMatrix; }

    void setDistortionCoefficients(const cv::Matx15d& _distortionCoefficients) {
        distortionCoefficients = _distortionCoefficients;
    }
    cv::Matx15d getDistortionCoefficients() const {
        return distortionCoefficients;
    }

    std::vector<cv::Matx34d>& getExtrinsicMatrixList() {
        return ExtrinsicMatrixList;
    }
    const std::vector<cv::Matx34d>& getExtrinsicMatrixList() const {
        return ExtrinsicMatrixList;
    }

    std::vector<cv::Mat>& getImageList() { return imageList; }
    const std::vector<cv::Mat>& getImageList() const { return imageList; }

    std::vector<std::pair<cv::Point3f, cv::Vec3b>>& getPointCloud() {
        return pointCloud;
    }
    const std::vector<std::pair<cv::Point3f, cv::Vec3b>>& getPointCloud()
        const {
        return pointCloud;
    }
};

/*************************************************************************************************/
