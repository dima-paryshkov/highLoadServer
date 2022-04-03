# high Load Server with pool threads

Development of a highly loaded multi-process server. The server receives many requests from clients, the execution of each request involves expensive calculations. The server consists of a main process that accepts requests from clients and sends results back to them, and worker processes that perform the calculations themselves. The master process distributes work among the workers, dynamic load balancing is also carried out.

Implementation:
Server - the server accepts the client, receives data from it and adds the task to the thread pool. The task is to raise the matrix to a given power. The matrix size does not exceed 128x128. After the task has been completed, the server sends the solution to the client. The client then exits.

Compile:

    make all

or

	  g++ server.cpp -o server -lpthread
	  g++ client.cpp -o client -lpthread
    
Launch:

    ./server
    ./client <IP addres>

`<IP addres> - IPv4. You can use localhost: 127.0.0.1
