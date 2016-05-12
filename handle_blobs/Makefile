CC=g++
SONJACC=clang++ -stdlib=libc++
CVFLAGS=`pkg-config --cflags opencv`
CVLIBS=`pkg-config --libs opencv`
MACBOOSTFLAGS=-lboost_thread-mt -lboost_system
BOOSTFLAGS=-lboost_thread -lboost_system

mac:
	$(CC) main.cc -o handle_blobs $(MACBOOSTFLAGS) $(CVLIBS) $(CVFLAGS)
linux:
	$(CC) main.cc -o handle_blobs $(BOOSTFLAGS) $(CVLIBS) $(CVFLAGS)
sonja:
	$(SONJACC) main.cc -o handle_blobs $(MACBOOSTFLAGS) $(CVLIBS) $(CVFLAGS)

clean:
	rm handle_blobs
