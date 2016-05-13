from mat_from_c import CMatProcess
import argparse
import cv2

def processData(f, cProcessor):
	result = cProcessor.processPaths([f], 'background_corr')

	i = 0
	while True:
		cv2.imshow('image', cv2.pyrUp(result[i]))
		cv2.waitKey(20)
		i = i + 1
		i = i % len(result)

	cv2.destroyAllWindows()


def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('input', help='Source of data (movies)')

	argparser.set_defaults(skip=2)
	args = argparser.parse_args()

	cProcessor = CMatProcess()

	processData(args.input, cProcessor)

if __name__ == "__main__":
    main()