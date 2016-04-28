CC=g++
CVFLAGS=`pkg-config --cflags opencv`
CVLIBS=`pkg-config --libs opencv`
MACBOOSTFLAGS=-lboost_thread-mt -lboost_system
BOOSTFLAGS=-lboost_thread -lboost_system

mac:
	$(CC) main.cc -o vidprocess $(MACBOOSTFLAGS) $(CVLIBS) $(CVFLAGS)
linux:
	$(CC) main.cc -o vidprocess $(BOOSTFLAGS) $(CVLIBS) $(CVFLAGS)

clean:
	rm vidprocess
