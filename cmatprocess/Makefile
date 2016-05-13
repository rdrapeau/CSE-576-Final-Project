CC=g++ -std=c++11
SONJACC=clang++ -stdlib=libc++
CVFLAGS=`pkg-config --cflags opencv`
CVLIBS=`pkg-config --libs opencv`
MACBOOSTFLAGS=-lboost_thread-mt -lboost_system
BOOSTFLAGS=-lboost_thread -lboost_system

mac:
	$(CC) main.cc -o cmatprocess $(MACBOOSTFLAGS) $(CVLIBS) $(CVFLAGS)
linux:
	$(CC) main.cc -o cmatprocess $(BOOSTFLAGS) $(CVLIBS) $(CVFLAGS)
sonja:
	$(SONJACC) main.cc -o cmatprocess $(MACBOOSTFLAGS) $(CVLIBS) $(CVFLAGS)

clean:
	rm cmatprocess
