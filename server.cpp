#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace std;

struct request
{   
    int sockfd;
    struct sockaddr_in cliaddr;
    float* matrix;
};

vector<request> queue;

int totalNumberOfConnection = 0;
int currentNumberOfConnection = 0;

const int numberOfThread = 11;

void* matrixMultiplication(void* arg)
{

}


int main()
{
    pthread_t thread[numberOfThread];
    // descriptor for listen socket
    int sockfd;


}
