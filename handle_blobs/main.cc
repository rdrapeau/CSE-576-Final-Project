#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <stdio.h>

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

static void processFrameChunk(
        vector<Mat> *frames,
        int start, int end) {
    // cout << "Processing chunk from " << start << " to " << end << endl;

    for (int i = start; i <= end; i++) {
        Mat frame = (*frames)[i];

        Mat combined = colorVarianceImage(frame) + gradientMagnitudeImage(frame);
        Mat normalized;
        normalize(combined, normalized, 0, 255, NORM_MINMAX, CV_8UC1);

        Mat thresholded;
        adaptiveThreshold(normalized, thresholded, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, -40);

        // Set result
        (*frames)[i] = thresholded;
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
    cout << programName << " infile cores" << endl;
    cout << "\tVideos, and images are supported." << endl;
    cout << "\tCores is the number of parallel threads activated for processing video frames. It must be >0" << endl;
    exit(1);
}

static void outputFrame(Mat in) {
    cout << '{' << endl;
    unsigned char *input = (unsigned char*)(in.data);
    for(int j = 0; j < in.rows; j++){
        for(int i = 0; i < in.cols; i++){
            unsigned char val = input[in.step * j + i ];
            if (val > 100) {
                cout << i << endl;
                cout << j << endl;
            }
        }
    }
    cout << '}' << endl;

    // namedWindow( "Display window", WINDOW_AUTOSIZE );
    // imshow( "Display window", in );
    // waitKey(0);
}

int main(int argc, char** argv) {
    // Process command line arguments
    if (argc != 3) usage(argv[0]);

    int coreCount = atoi(argv[2]);
    if (coreCount <= 0) usage(argv[0]);

    string inFilePath(argv[1]);


    // Open video
    bool video = inFilePath.find("mp4") != string::npos || inFilePath.find("mov") !=string::npos;

    vector<Mat> frames;
    VideoCapture inputVideo(inFilePath);
    Size S;
    if (video) {
        if(!inputVideo.isOpened())
            return -1;

        S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),
                      (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
        // cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
        //     << " of nr#: " << inputVideo.get(CV_CAP_PROP_FRAME_COUNT) << endl;

        // cout << "Loading video into memory..." << endl;
        frames = getFrames(inputVideo);
    } else {
        frames.push_back(imread(inFilePath, CV_LOAD_IMAGE_COLOR));
    }

    // cout << "Loaded " << frames.size() << " frames" << endl;
    // cout << "Processing frames..." << endl;

    if (video) {
        // Process each frame
        int chunkSize = (frames.size() / coreCount);

        // Use a thread pool to watch over all the threads
        boost::thread_group group;
        int start = 0;
        for (int i = 0; i < coreCount - 1; i++) {
            // Calculate chunk bounds
            int end = start + chunkSize;

            // Kick off the thread
            boost::thread *processFrameJob = new boost::thread(processFrameChunk, &frames, start, end);
            group.add_thread(processFrameJob);

            start = end + 1;
        }

        int end = frames.size() - 1;
        boost::thread *processFrameJob = new boost::thread(processFrameChunk, &frames, start, end);
        group.add_thread(processFrameJob);

        // Will destruct thread objects
        group.join_all();
    } else {
        processFrameChunk(&frames, 0, 0);
    }

    // cout << "Outputing frames..." << endl;

    for (int i = 0; i < frames.size(); i++) {
        outputFrame(frames[i]);
    }


    return 0;
}
