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
#include <stdlib.h>
#include <malloc.h>

using namespace std;

struct request
{
    int sockfd;
    struct sockaddr_in cliaddr;
    socklen_t len;
    float *matrix;
    int size;
    int degree;
};

vector<request> queue;

int totalNumberOfConnection = 0;
int currentNumberOfConnection = 0;

const int numberOfThread = 11;

void *matrixMultiplication(void *arg)
{
}

int main()
{
    pthread_t thread[numberOfThread];

    for (int i = 0; i < numberOfThread; i++)
    {
        if (pthread_create(&thread[i], (pthread_attr_t *)NULL, matrixMultiplication, (void *)NULL) == -1)
        {
            fprintf(stderr, "An error occurred while creating the %d thread: %s", i + 1, strerror(errno));
            exit(-1);
        }
    }

    /* descriptor for listen socket */
    int sockfd;

    /* create TCP-socket */
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        fprintf(stderr, "Can't create listen socket: %s", strerror(errno));
        exit(-1);
    }

    /* struct for full addres of server */
    struct sockaddr_in servaddr;

    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(51000);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        fprintf(stderr, "Can't bind socket: %s", strerror(errno));
        close(sockfd);
        exit(-1);
    }

    if (listen(sockfd, 50) < 0)
    {
        fprintf(stderr, "Can't listen socket: %s", strerror(errno));
        close(sockfd);
        exit(-1);
    }

    /* create second process for send information */
    int pid = fork();
    if (pid == -1)
    {
        fprintf(stderr, "Can't create second process (fork): %s", strerror(errno));
        close(sockfd);
        exit(-1);
    }
    else if (pid == 0)
    {
        while (1)
        {
            request rq;
            if ((rq.sockfd = accept(sockfd, (struct sockaddr *)&rq.cliaddr, &rq.len)) < 0)
            {
                perror("Can't accept new socket");
                close(sockfd);
                exit(-1);
            }

            totalNumberOfConnection++;
            currentNumberOfConnection++;
            
            if (read(rq.sockfd, (void*)rq.size, sizeof(int)) < 0)
            {
                perror("Can't read size from socket");
                close(sockfd);
                exit(-1);
            }

            if (read(rq.sockfd, (void*)rq.degree, sizeof(int)) < 0)
            {
                perror("Can't read degree from socket");
                close(sockfd);
                exit(-1);
            }

            rq.matrix = (float*)malloc(rq.size * rq.size * sizeof(float));

            if (read(rq.sockfd, (void*)rq.matrix, rq.size) < 0)
            {
                perror("Can't read size from socket");
                close(sockfd);
                exit(-1);
            }

            queue.push_back(rq);
        }
    }
    else
    {
        while (1)
        {
        }
    }
}
