all: server.cpp client.cpp
	g++ server.cpp -o server -lpthread
	g++ client.cpp -o client -lpthread

client: client.cpp
	g++ client.cpp -o client -lpthread

server: server.cpp
	g++ server.cpp -o server -lpthread