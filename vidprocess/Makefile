CC=g++
CVFLAGS=`pkg-config --cflags opencv`
CVLIBS=`pkg-config --libs opencv`
BOOSTFLAGS=-lboost_thread -lboost_system


executable:
	$(CC) main.cc -o vidprocess $(BOOSTFLAGS) $(CVLIBS) $(CVFLAGS)

clean:
	rm vidprocess
