#include "structure_from_motion.hpp"

#define DEBUG

template <class Point>
void alignPoints(const std::vector<Point>& point1,
                 const std::vector<Point>& point2,
                 const std::vector<cv::DMatch>& matches,
                 std::vector<Point>& points1Matched,
                 std::vector<Point>& points2Matched,
                 std::vector<index_type>& points1Indices,
                 std::vector<index_type>& points2Indices) {
    points1Matched.clear();
    points2Matched.clear();
    points1Indices.clear();
    points2Indices.clear();

    for (const auto& match : matches) {
        points1Matched.push_back(point1[match.queryIdx]);
        points2Matched.push_back(point2[match.trainIdx]);
        points1Indices.push_back(match.queryIdx);
        points2Indices.push_back(match.trainIdx);
    }
}

void keyPointsToPoints(const std::vector<cv::KeyPoint>& keyPoints,
                       std::vector<cv::Point2f>& points) {
    points.clear();

    for (const auto& keyPoint : keyPoints) {
        points.push_back(keyPoint.pt);
    }
}

void flipMatches(const std::vector<cv::DMatch>& matches1,
                 std::vector<cv::DMatch>& matches2) {
    matches2.clear();

    for (const auto& match1 : matches1) {
        auto match2(match1);
        std::swap(match2.queryIdx, match2.trainIdx);

        matches2.push_back(match2);
    }
}

void StructureFromMotion::loadImage(const boost::filesystem::path& filePath) {
    imageList.emplace_back(cv::imread(filePath.string()));
}

void StructureFromMotion::loadParameter(
    const boost::filesystem::path& filePath) {
    boost::property_tree::ptree parameters;

    boost::property_tree::read_xml(filePath.string(), parameters);

    std::vector<double> intrinsicMatrixValues;
    for (const auto& value : parameters.get_child("intrinsic_matrix")) {
        intrinsicMatrixValues.push_back(
            boost::lexical_cast<double>(value.second.data()));
    }

    std::vector<double> distortionCoefficientsValues;
    for (const auto& value : parameters.get_child("distortion_coefficients")) {
        distortionCoefficientsValues.push_back(
            boost::lexical_cast<double>(value.second.data()));
    }

    IntrinsicMatrix = cv::Matx33d(intrinsicMatrixValues.data());
    distortionCoefficients = cv::Matx15d(distortionCoefficientsValues.data());
}

void StructureFromMotion::process() {
    initialize();
    detectFeatures();
    matchFeatures();
    reconstructFoundation();
    reconstructMoreViews();
}

void StructureFromMotion::initialize() {
    keyPointsList.resize(imageList.size());
    pointsList.resize(imageList.size());
    descriptorsList.resize(imageList.size());
    matchesMatrix.resize(boost::extents[imageList.size()][imageList.size()]);
    ExtrinsicMatrixList.resize(imageList.size());

    for (index_type view(0); view < imageList.size(); ++view) {
        viewsNotProcessed.insert(view);
    }
}

void StructureFromMotion::detectFeatures() {
#ifdef DEBUG

    std::cout << "===================detecting features===================="
              << std::endl;

#endif  // DEBUG

    std::vector<std::thread> threads;
    std::mutex mtx;

    index_type numThreads(std::thread::hardware_concurrency());

    index_type view(0);
    for (index_type index(0); index < numThreads; ++index) {
        index_type num(
            static_cast<index_type>(imageList.size()) / numThreads +
            (index < (static_cast<index_type>(imageList.size()) % numThreads)));

        threads.emplace_back(
            [&](const auto& range) {
                for (index_type view(range.first); view < range.second;
                     ++view) {
                    const auto& image(imageList[view]);

                    auto detector(cv::xfeatures2d::SURF::create());

                    std::vector<cv::KeyPoint> keyPoints;
                    detector->detect(image, keyPoints);

                    std::vector<cv::Point2f> points;
                    keyPointsToPoints(keyPoints, points);

                    cv::Mat descriptors;
                    detector->compute(image, keyPoints, descriptors);

#ifdef DEBUG

                    mtx.lock();
                    std::cout << "(" << view << "): " << keyPoints.size()
                              << " keypoints detected" << std::endl;
                    mtx.unlock();

#endif  // DEBUG

                    keyPointsList[view] = move(keyPoints);
                    pointsList[view] = move(points);
                    descriptorsList[view] = std::move(descriptors);
                }
            },
            std::make_pair(view, view + num));

        view += num;
    }

    for (auto& thread : threads) thread.join();
}

void StructureFromMotion::matchFeatures() {
#ifdef DEBUG

    std::cout << "====================matching features===================="
              << std::endl;

#endif  // DEBUG

    std::vector<std::thread> threads;
    std::mutex mtx;

    index_type numThreads(std::thread::hardware_concurrency());

    std::vector<index_type> numCombinations;
    for (index_type view(0); view < imageList.size(); ++view) {
        numCombinations.push_back(
            numCombinations.empty() ? 0 : numCombinations.back() + view);
    }

    index_type combination(0);
    for (index_type index(0); index < numThreads; ++index) {
        index_type num(numCombinations.back() / numThreads +
                       (index < (numCombinations.back() % numThreads)));

        threads.emplace_back(
            [&](const auto& range) {
                for (index_type combination(range.first);
                     combination < range.second; ++combination) {
                    index_type view1(static_cast<index_type>(distance(
                        begin(numCombinations),
                        lower_bound(begin(numCombinations),
                                    end(numCombinations), combination + 1))));
                    index_type view2(combination - (view1 - 1) * view1 / 2);

                    const auto& descriptors1(descriptorsList[view1]);
                    const auto& descriptors2(descriptorsList[view2]);

                    auto matcher(cv::DescriptorMatcher::create("FlannBased"));

                    std::vector<std::vector<cv::DMatch>> matches;
                    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

                    std::vector<cv::DMatch> prunedMatches1;

                    for (const auto& match : matches) {
                        if (match[0].distance < 0.8 * match[1].distance) {
                            prunedMatches1.push_back(match[0]);
                        }
                    }

                    std::vector<cv::DMatch> prunedMatches2;
                    flipMatches(prunedMatches1, prunedMatches2);

#ifdef DEBUG

                    mtx.lock();
                    std::cout << "(" << view1 << ", " << view2
                              << "): " << prunedMatches1.size()
                              << " keypoints matched" << std::endl;
                    mtx.unlock();

#endif  // DEBUG

                    matchesMatrix[view1][view2] = move(prunedMatches1);
                    matchesMatrix[view2][view1] = move(prunedMatches2);
                }
            },
            std::make_pair(combination, combination + num));

        combination += num;
    }

    for (auto& thread : threads) thread.join();
}

void StructureFromMotion::reconstructFoundation() {
#ifdef DEBUG

    std::cout << "================reconstructing foundation================"
              << std::endl;

#endif  // DEBUG

    hash_map<std::pair<index_type, index_type>,
             std::tuple<std::vector<cv::DMatch>, std::vector<cv::DMatch>,
                        cv::Matx34d, cv::Matx34d>,
             Hash<std::pair<index_type, index_type>>>
        viewPairs;

    for (index_type view1(0); view1 < imageList.size(); ++view1) {
        for (index_type view2(view1 + 1); view2 < imageList.size(); ++view2) {
            std::vector<cv::DMatch> prunedMatches1;
            std::vector<cv::DMatch> prunedMatches2;
            cv::Matx34d ExtrinsicMatrix1;
            cv::Matx34d ExtrinsicMatrix2;

            findExtrinsicMatrices(view1, view2, prunedMatches1, prunedMatches2,
                                  ExtrinsicMatrix1, ExtrinsicMatrix2);

            viewPairs.emplace(
                std::piecewise_construct, std::forward_as_tuple(view1, view2),
                std::forward_as_tuple(
                    move(prunedMatches1), std::move(prunedMatches2),
                    std::move(ExtrinsicMatrix1), std::move(ExtrinsicMatrix2)));
        }
    }

    auto maxElement(max_element(
        begin(viewPairs), end(viewPairs),
        [&](const auto& viewPair1, const auto& viewPair2) {
            return static_cast<float>(std::get<0>(viewPair1.second).size()) /
                       matchesMatrix[viewPair1.first.first]
                                    [viewPair1.first.second]
                                        .size() <
                   static_cast<float>(std::get<0>(viewPair2.second).size()) /
                       matchesMatrix[viewPair2.first.first]
                                    [viewPair2.first.second]
                                        .size();
        }));

    matchesMatrix[maxElement->first.first][maxElement->first.second] =
        std::get<0>(maxElement->second);
    matchesMatrix[maxElement->first.second][maxElement->first.first] =
        std::get<1>(maxElement->second);
    ExtrinsicMatrixList[maxElement->first.first] =
        std::get<2>(maxElement->second);
    ExtrinsicMatrixList[maxElement->first.second] =
        std::get<3>(maxElement->second);

    triangulateViews(maxElement->first.first, maxElement->first.second);

#ifdef DEBUG

    std::cout << "(" << maxElement->first.first << ", "
              << maxElement->first.second << "): triangulated" << std::endl;

#endif  // DEBUG

    viewsNotProcessed.erase(maxElement->first.first);
    viewsNotProcessed.erase(maxElement->first.second);

    viewsProcessed.insert(maxElement->first.first);
    viewsProcessed.insert(maxElement->first.second);
}

void StructureFromMotion::reconstructMoreViews() {
#ifdef DEBUG

    std::cout << "================reconstructing more views================"
              << std::endl;

#endif  // DEBUG

    while (!viewsNotProcessed.empty()) {
        hash_map<index_type,
                 std::pair<std::vector<cv::Point3f>, std::vector<cv::Point2f>>>
            correspondences3d2d;

        for (auto view : viewsNotProcessed) {
            std::vector<cv::Point3f> points3d;
            std::vector<cv::Point2f> points2d;

            for (index_type index(0); index < pointCloud.size(); ++index) {
                for (const auto& viewAndPoint : matches3d2d[index]) {
                    const auto& matches(
                        matchesMatrix[view][viewAndPoint.first]);

                    auto it(find_if(
                        begin(matches), end(matches), [&](const auto& match) {
                            return match.trainIdx == viewAndPoint.second;
                        }));

                    if (it != end(matches)) {
                        points3d.push_back(pointCloud[index].first);
                        points2d.push_back(pointsList[view][it->queryIdx]);

                        break;
                    }
                }
            }

            correspondences3d2d.emplace(
                std::piecewise_construct, std::forward_as_tuple(view),
                std::forward_as_tuple(points3d, points2d));
        }

        auto maxElement(std::max_element(
            std::begin(correspondences3d2d), std::end(correspondences3d2d),
            [](const auto& correspondence1, const auto& correspondence2) {
                return correspondence1.second.first.size() <
                       correspondence2.second.first.size();
            }));

        cv::Matx31d rotation;
        cv::Matx31d translation;

        cv::solvePnPRansac(maxElement->second.first, maxElement->second.second,
                           IntrinsicMatrix, distortionCoefficients, rotation,
                           translation, false, 1000, 8.0, 0.99);

        cv::Matx33d Rotation;
        cv::Rodrigues(rotation, Rotation);

        cv::Matx34d ExtrinsicMatrix(
            Rotation(0, 0), Rotation(0, 1), Rotation(0, 2), translation(0),
            Rotation(1, 0), Rotation(1, 1), Rotation(1, 2), translation(1),
            Rotation(2, 0), Rotation(2, 1), Rotation(2, 2), translation(2));

        ExtrinsicMatrixList[maxElement->first] = std::move(ExtrinsicMatrix);

        for (auto view : viewsProcessed) {
            std::vector<cv::DMatch> prunedMatches1;
            std::vector<cv::DMatch> prunedMatches2;
            cv::Matx34d ExtrinsicMatrix1;
            cv::Matx34d ExtrinsicMatrix2;

            findExtrinsicMatrices(maxElement->first, view, prunedMatches1,
                                  prunedMatches2, ExtrinsicMatrix1,
                                  ExtrinsicMatrix2);

            matchesMatrix[maxElement->first][view] = prunedMatches1;
            matchesMatrix[view][maxElement->first] = prunedMatches2;

            triangulateViews(maxElement->first, view);

#ifdef DEBUG

            std::cout << "(" << maxElement->first << ", " << view
                      << "): triangulated" << std::endl;

#endif  // DEBUG
        }

        viewsNotProcessed.erase(maxElement->first);
        viewsProcessed.insert(maxElement->first);
    }
}

void StructureFromMotion::findExtrinsicMatrices(
    index_type view1, index_type view2, std::vector<cv::DMatch>& prunedMatches1,
    std::vector<cv::DMatch>& prunedMatches2, cv::Matx34d& ExtrinsicMatrix1,
    cv::Matx34d& ExtrinsicMatrix2) {
    std::vector<cv::Point2f> points1Matched;
    std::vector<cv::Point2f> points2Matched;
    std::vector<index_type> points1Indices;
    std::vector<index_type> points2Indices;

    alignPoints(pointsList[view1], pointsList[view2],
                matchesMatrix[view1][view2], points1Matched, points2Matched,
                points1Indices, points2Indices);

    cv::Mat inliers;

    cv::Mat EssentialMatrix(findEssentialMat(points1Matched, points2Matched,
                                             IntrinsicMatrix, cv::RANSAC, 0.999,
                                             1.0, inliers));

    cv::Matx33d Rotation;
    cv::Matx31d translation;

    cv::recoverPose(EssentialMatrix, points1Matched, points2Matched,
                    IntrinsicMatrix, Rotation, translation, inliers);

    ExtrinsicMatrix1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0;

    ExtrinsicMatrix2 << Rotation(0, 0), Rotation(0, 1), Rotation(0, 2),
        translation(0), Rotation(1, 0), Rotation(1, 1), Rotation(1, 2),
        translation(1), Rotation(2, 0), Rotation(2, 1), Rotation(2, 2),
        translation(2);

    for (index_type index(0); index < inliers.rows; index++) {
        if (inliers.at<uchar>(index)) {
            prunedMatches1.push_back(matchesMatrix[view1][view2][index]);
            prunedMatches2.push_back(matchesMatrix[view2][view1][index]);
        }
    }
}

void StructureFromMotion::triangulateViews(index_type view1, index_type view2) {
    std::vector<cv::Point2f> points1Matched;
    std::vector<cv::Point2f> points2Matched;
    std::vector<index_type> points1Indices;
    std::vector<index_type> points2Indices;

    alignPoints(pointsList[view1], pointsList[view2],
                matchesMatrix[view1][view2], points1Matched, points2Matched,
                points1Indices, points2Indices);

    cv::Mat points3dHomogeneous;
    cv::triangulatePoints(IntrinsicMatrix * ExtrinsicMatrixList[view1],
                          IntrinsicMatrix * ExtrinsicMatrixList[view2],
                          points1Matched, points2Matched, points3dHomogeneous);

    std::vector<cv::Point3f> points3d;
    convertPointsFromHomogeneous(cv::Mat(points3dHomogeneous.t()).reshape(4),
                                 points3d);

    cv::Matx31d rotation1;
    cv::Rodrigues(ExtrinsicMatrixList[view1].get_minor<3, 3>(0, 0), rotation1);

    cv::Matx31d translation1(ExtrinsicMatrixList[view1].get_minor<3, 1>(0, 3));

    std::vector<cv::Point2f> points1Projected;
    cv::projectPoints(points3d, rotation1, translation1, IntrinsicMatrix,
                      cv::Mat(), points1Projected);
    // projectPoints(points3d, rvec1, tvec1, IntrinsicMatrix,
    // distortionCoefficients, points1Projected);

    cv::Matx31d rotation2;
    cv::Rodrigues(ExtrinsicMatrixList[view2].get_minor<3, 3>(0, 0), rotation2);

    cv::Matx31d translation2(ExtrinsicMatrixList[view2].get_minor<3, 1>(0, 3));

    std::vector<cv::Point2f> points2Projected;
    cv::projectPoints(points3d, rotation2, translation2, IntrinsicMatrix,
                      cv::Mat(), points2Projected);
    // projectPoints(points3d, rvec2, tvec2, IntrinsicMatrix,
    // distortionCoefficients, points2Projected);

    for (index_type index(0); index < points3d.size(); index++) {
        if (norm(points1Projected[index] - points1Matched[index]) > 8.0 ||
            norm(points2Projected[index] - points2Matched[index]) > 8.0)
            continue;

        const auto& color(imageList[view1].at<cv::Vec3b>(
            pointsList[view1][points1Indices[index]]));

        pointCloud.emplace_back(points3d[index], color);

        matches3d2d.push_back(
            {{view1, points1Indices[index]}, {view2, points2Indices[index]}});
    }

    signal(pointCloud);
}
