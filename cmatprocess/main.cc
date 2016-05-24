#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <stdio.h>
#include <iostream>
#include <functional>

using namespace cv;
using namespace std;

static Mat gradientMagnitudeImage(Mat in) {
    // Turn to greyscale
    Mat grey;
    cvtColor(in, grey, CV_BGR2GRAY);

    // Generate gradients
    Mat gradX, gradY;
    Mat absGradX, absGradY;
    Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // Gradient X
    Sobel(grey, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(gradX, absGradX);

    // Gradient Y
    Sobel(grey, gradY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(gradY, absGradY);

    // Total Gradient (approximate)
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);

    Mat blurred;
    GaussianBlur(grad, blurred, Size(15, 15), 0, 0);

    Mat result;
    normalize(blurred, result, 0, 1, NORM_MINMAX, CV_32F);
    return result;
}

static Mat colorVarianceImage(Mat in) {
    // Turn to greyscale
    Mat grey;
    cvtColor(in, grey, CV_BGR2GRAY);

    // Blur image
    Mat greyBlurred;
    GaussianBlur(grey, greyBlurred, Size(21, 21), 0, 0);

    // Split into channels
    Mat bgr[3];
    split(in, bgr);

    // Take difference from each pixel to the average and sum
    Mat diffB;
    absdiff(bgr[0], greyBlurred, diffB);
    Mat diffG;
    absdiff(bgr[1], greyBlurred, diffG);
    Mat diffR;
    absdiff(bgr[2], greyBlurred, diffR);

    Mat diffBlurred;
    GaussianBlur(diffR + diffG + diffB, diffBlurred, Size(15, 15), 0, 0);

    Mat result;
    normalize(diffBlurred, result, 0, 1, NORM_MINMAX, CV_32F);
    return result;
}


static Mat handleBlobs(Mat frame) {
    Mat combined = colorVarianceImage(frame) + gradientMagnitudeImage(frame);
    Mat normalized;
    normalize(combined, normalized, 0, 255, NORM_MINMAX, CV_8UC1);
    return normalized;
}

static int counter = 0;
static int DOWN_SAMPLE = 8;
static vector<Mat> computeDescriptors(Mat img) {
    int SIZE = 11;

    int rows = img.rows;
    int cols = img.cols;
    int sizes[] = {SIZE * SIZE};
    vector<Mat> vecs;

    for (int i = 0; i < img.rows; i += DOWN_SAMPLE)
        for (int j = 0; j < img.cols; j += DOWN_SAMPLE) {
            int vecIndex = 0;

            Mat vec = Mat(1, sizes, CV_32FC1, Scalar(0));

            for (int ki = -SIZE / 2; ki <= SIZE / 2; ki++) {
                for (int kj = -SIZE / 2; kj <= SIZE / 2; kj++) {
                    float refVal = img.at<uchar>(i, j);

                    float val = 0;
                    if (i + ki >= 0 && i + ki < img.rows &&
                        j + kj >= 0 && j + kj < img.cols) {
                        val = img.at<uchar>(i + ki, j + kj);
                    }

                    vec.at<float>(vecIndex) = refVal;
                    vecIndex++;
                }
            }

            vecs.push_back(vec);
        }

    return vecs;
}


static vector<Mat> backgroundSubtract_bgDescriptor;
static Mat backgroundSubtract(Mat foreground) {
    // cout << "subtracting bg descriptors.. " << endl;
    Mat grey;
    cvtColor(foreground, grey, CV_BGR2GRAY);
    Mat foregroundBlurred;
    GaussianBlur(grey, foregroundBlurred, Size(5, 5), 0, 0);

    vector<Mat> descriptor = computeDescriptors(foregroundBlurred);

    // cout << "done.. Comparing images " << endl;

    Mat result(foreground.rows / DOWN_SAMPLE, foreground.cols / DOWN_SAMPLE, CV_32F, Scalar(0));
    int pixIndex = 0;
    for (int i = 0; i < result.rows; i++)
        for (int j = 0; j < result.cols; j++) {

            Mat vec1 = descriptor[pixIndex];
            Mat vec2 = backgroundSubtract_bgDescriptor[pixIndex];
            Mat sub = (vec1 - vec2);

            float val = sqrt(sub.dot(sub));

            result.at<float>(i, j) = val;
            pixIndex++;
        }

    Mat normed;
    normalize(result, normed, 0, 255, NORM_MINMAX, CV_8UC1);

    // Mat thresholded;
    // adaptiveThreshold(normed, thresholded, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, -40);

    return normed;
}

static vector<Mat> extractPatches(Mat img, int size) {
    vector<Mat> result;
    for (int i = size; i < img.rows; i += size)
        for (int j = size; j < img.cols; j += size) {
            Rect area(j - size, i - size, size - 1, size - 1);
            // cout << "rows: " << img.rows << " cols:" << img.cols << endl;
            // cout << "i: " << i << " j: " << j << " a: " << area.x << "," << area.y << "," << area.width << "," << area.height << endl;
            result.push_back(img(area));
        }

    return result;
}

static vector<Mat> bgPatches;
static int PATCH_SIZE = 17;
static Mat backgroundCorrelation(Mat foreground) {
    Mat foregroundBlurred;
    GaussianBlur(foreground, foregroundBlurred, Size(5, 5), 0, 0);
    Mat foregroundShifted;
    pyrMeanShiftFiltering(foregroundBlurred, foregroundShifted, 10, 20, 3);

    vector<Mat> fgPatches = extractPatches(foregroundShifted, PATCH_SIZE);

    Mat result(foreground.rows / PATCH_SIZE, foreground.cols / PATCH_SIZE, CV_32F, Scalar(0));
    int pixIndex = 0;
    for (int i = 0; i < result.rows; i++)
        for (int j = 0; j < result.cols; j++) {

            Mat patch1 = bgPatches[pixIndex];
            Mat patch2 = fgPatches[pixIndex];
            Mat moddedPatch1;
            Mat moddedPatch2;

            cvtColor(patch1, moddedPatch1, CV_BGR2HSV);
            cvtColor(patch2, moddedPatch2, CV_BGR2HSV);

            Mat diffImage;
            absdiff(moddedPatch1, moddedPatch2, diffImage);


            float dist = 0;
            for (int pj=0; pj<diffImage.rows; ++pj)
                for (int pi=0; pi<diffImage.cols; ++pi) {
                    cv::Vec3b pix = diffImage.at<cv::Vec3b>(pj,pi);

                    float diffVal = (pix[0]*pix[0] + pix[1]*pix[1] + pix[2]*pix[2]);
                    dist += diffVal;
                }

            // Mat resultCorr(1, 1, CV_32FC1);

            // /// Do the Matching and Normalize
            // matchTemplate(patch1, patch2, resultCorr, CV_TM_SQDIFF);

            // float val = resultCorr.at<float>(0, 0);
            result.at<float>(i, j) = dist;
            pixIndex++;
        }

    Mat normed;
    normalize(result, normed, 0, 255, NORM_MINMAX, CV_8UC1);

    return normed;
}

static void mapFrameChunk(
        vector<Mat> *frames,
        int start, int end,
        Mat (*f)(Mat)
        ) {
    for (int i = start; i <= end; i++) {
        Mat frame = (*frames)[i];
        Mat result = f(frame);
        (*frames)[i] = result;
    }
}

static vector<Mat> getFrames(VideoCapture &video) {
    vector<Mat> result;
    int currentFrame = 0;
    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) break;
        result.push_back(frame);
        // cout << "Frame " << (currentFrame) << " loaded" << endl;
        currentFrame++;
    }

    return result;
}

static void usage(char* programName) {
    cout << "Usage: " << endl;
    cout << programName << " cores" << endl;
    exit(1);
}

static void outputFrame(Mat in) {
    string name = tmpnam(nullptr);
    name += ".png";
    imwrite(name, in);
    cout << name << endl;
}

static void mapFrames(vector<vector<Mat>> &fileFrames, int coreCount, Mat (*f)(Mat)) {
    for (auto &frames: fileFrames) {
        if (frames.size() > 1) {
            // Process each frame
            int chunkSize = (frames.size() / coreCount);

            // Use a thread pool to watch over all the threads
            boost::thread_group group;
            int start = 0;
            for (int i = 0; i < coreCount - 1; i++) {
                // Calculate chunk bounds
                int end = start + chunkSize;

                // Kick off the thread
                boost::thread *processFrameJob = new boost::thread(mapFrameChunk, &frames, start, end, f);
                group.add_thread(processFrameJob);

                start = end + 1;
            }

            int end = frames.size() - 1;
            boost::thread *processFrameJob = new boost::thread(mapFrameChunk, &frames, start, end, f);
            group.add_thread(processFrameJob);

            // Will destruct thread objects
            group.join_all();
        } else {
            mapFrameChunk(&frames, 0, 0, f);
        }
    }
}

static vector<Mat> processInput(vector<string> inFiles, string functionName, int coreCount) {
    vector<vector<Mat>> fileFrames;
    for (auto const& inFilePath: inFiles) {
        // Open video
        bool video = inFilePath.find("mp4") != string::npos || inFilePath.find("mov") !=string::npos;
        vector<Mat> frames;

        VideoCapture inputVideo(inFilePath);
        Size S;
        if (video) {
            if(!inputVideo.isOpened())
                return frames;

            S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),
                          (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
            frames = getFrames(inputVideo);
        } else {
            frames.push_back(imread(inFilePath, CV_LOAD_IMAGE_COLOR));
        }

        fileFrames.push_back(frames);
    }

    if (functionName == "handle_blobs") {
        mapFrames(fileFrames, coreCount, &handleBlobs);
    } else if (functionName == "background_subtract") {
        Mat grey;
        cvtColor(fileFrames[0][0], grey, CV_BGR2GRAY);
        Mat backgroundBlurred;
        GaussianBlur(grey, backgroundBlurred, Size(5, 5), 0, 0);
        backgroundSubtract_bgDescriptor = computeDescriptors(backgroundBlurred);
        mapFrames(fileFrames, coreCount, &backgroundSubtract);
    } else if (functionName == "background_corr") {
        Mat backgroundBlurred;
        GaussianBlur(fileFrames[0][0], backgroundBlurred, Size(5, 5), 0, 0);
        Mat backgroundShifted;
        pyrMeanShiftFiltering(backgroundBlurred, backgroundShifted, 10, 20, 3);
        bgPatches = extractPatches(backgroundShifted, PATCH_SIZE);
        mapFrames(fileFrames, coreCount, &backgroundCorrelation);
    }

    vector<Mat> frames;
    for (auto const& toAdd: fileFrames) {
        frames.insert(frames.end(), toAdd.begin(), toAdd.end());
    }

    return frames;
}

int main(int argc, char** argv) {
    // Process command line arguments
    if (argc != 2) usage(argv[0]);

    int coreCount = atoi(argv[1]);
    if (coreCount <= 0) usage(argv[0]);

    string buffer;
    while (1) {
        getline(cin, buffer);


        vector<string> strs;
        boost::split(strs, buffer, boost::is_any_of("\t "));
        vector<string> filePaths(strs.begin() + 1, strs.end());
        vector<Mat> frames = processInput(filePaths, strs[0], coreCount);

        cout << frames.size() << endl;
        for (int i = 0; i < frames.size(); i++) {
            outputFrame(frames[i]);
        }
    }


    return 0;
}
