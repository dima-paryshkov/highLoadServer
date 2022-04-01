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
    int *matrix;
    int *result;
    int size;
    int degree;
};

vector<request> queueTask;
vector<request> queueResolvedTask;

int totalNumberOfConnection = 0;
int currentNumberOfConnection = 0;

const int numberOfThread = 1;

pthread_mutex_t mutexQueueTask;
pthread_mutex_t mutexQueueResolvedTask;

/* descriptor for listen socket */
int sockfd;

bool active = true;

void *matrixMultiplication(void *arg)
{
    request rq;
    while (active)
    {
        sleep(1);
        if (pthread_mutex_lock(&mutexQueueTask) != 0)
        {
            perror("Can't lock mutex (child process)");
        }
        if (queueTask.size() > 0)
        {
            rq = queueTask[0];
            queueTask.pop_back();
            if (pthread_mutex_unlock(&mutexQueueTask) != 0)
            {
                perror("Can't unlock mutex (child process)");
            }
            rq.result = (int *)malloc(rq.size * rq.size * sizeof(int));
            for (int i = 0; i < rq.size; i++)
            {
                for (int j = 0; j < rq.size; j++)
                {
                    rq.result[i * rq.size] = 0;
                    for (int k = 0; k < rq.size; k++)
                    {
                        rq.result[i * rq.size + j] += rq.matrix[i * rq.size + k] + rq.matrix[j * rq.size + k];
                    }
                }
            }

            if (rq.degree > 2)
            {
                int *c = rq.result;
                rq.result = (int *)malloc(rq.size * rq.size * sizeof(int));
                for (int l = 2; l < rq.degree; l++)
                {
                    for (int i = 0; i < rq.size; i++)
                    {
                        for (int j = 0; j < rq.size; j++)
                        {
                            rq.result[i * rq.size + j] = 0;
                            for (int k = 0; k < rq.size; k++)
                            {
                                rq.result[i * rq.size + j] += c[i * rq.size + k] + rq.matrix[j * rq.size + k];
                            }
                        }
                    }
                    int* tmp = c;
                    c = rq.result;
                    rq.result = tmp;
                }
                rq.result = c;
                free(c);
            }

            if (pthread_mutex_lock(&mutexQueueResolvedTask) != 0)
            {
                perror("Can't lock mutex (child process)");
            }
            queueResolvedTask.push_back(rq);
            if (pthread_mutex_unlock(&mutexQueueResolvedTask) != 0)
            {
                perror("Can't unlock mutex (child process)");
            }
        }
        else if (pthread_mutex_unlock(&mutexQueueTask) != 0)
        {
            perror("Can't unlock mutex (child process)");
        }
    }
    pthread_exit(NULL);
}

void* sendmsg(void *arg)
{
    while (1)
    {
        if (queueResolvedTask.size() != 0)
        {
            int n;
            if ((n = write(queueResolvedTask[0].sockfd, queueResolvedTask[0].result, queueResolvedTask[0].size * queueResolvedTask[0].size)) < 0)
            {
                perror("Can't write information in socket (main process)");
                close(sockfd);
                exit(-1);
            }

            if (n < queueResolvedTask[0].size * queueResolvedTask[0].size)
            {
                fprintf(stderr, "Warning: not all information was write in socket (main process)\n");
            }

            close(queueResolvedTask[0].sockfd);

            if (pthread_mutex_lock(&mutexQueueResolvedTask) != 0)
            {
                perror("Can't lock mutex (main process)");
            }
            queueResolvedTask.pop_back();
            if (pthread_mutex_unlock(&mutexQueueResolvedTask) != 0)
            {
                perror("Can't unlock mutex (main process)");
            }
            currentNumberOfConnection--;
        }
    }
    pthread_exit(NULL);
}

int main()
{
    pthread_t thread[numberOfThread];
    pthread_t threadSend;

    if (pthread_create(&threadSend, (pthread_attr_t *)NULL, sendmsg, (void *)NULL) == -1)
    {
        perror("An error occurred while creating the send thread");
        exit(-1);
    }

    for (int i = 0; i < numberOfThread; i++)
    {
        if (pthread_create(&thread[i], (pthread_attr_t *)NULL, matrixMultiplication, (void *)NULL) == -1)
        {
            fprintf(stderr, "An error occurred while creating the %d thread: %s", i + 1, strerror(errno));
            exit(-1);
        }
    }

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

    // fprintf(stdout, "IP %s\n", inet_ntoa(servaddr.sin_addr));

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

    if (pthread_mutex_init(&mutexQueueTask, (pthread_mutexattr_t *)NULL) != 0)
    {
        perror("Can't init mutex (mutexQueueTask)");
        exit(-1);
    }

    if (pthread_mutex_init(&mutexQueueResolvedTask, (pthread_mutexattr_t *)NULL) != 0)
    {
        perror("Can't init mutex (mutexQueueResolvedTask)");
        exit(-1);
    }

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
        fprintf(stdout, "Accept new connection %d, IP %s\n", totalNumberOfConnection, inet_ntoa(rq.cliaddr.sin_addr));

        if (read(rq.sockfd, (void *)&rq.size, sizeof(int)) < 0)
        {
            perror("Can't read size from socket");
            close(sockfd);
            exit(-1);
        }

        if (read(rq.sockfd, (void *)&rq.degree, sizeof(int)) < 0)
        {
            perror("Can't read degree from socket");
            close(sockfd);
            exit(-1);
        }

        rq.matrix = (int *)malloc(rq.size * rq.size * sizeof(int));

        if (read(rq.sockfd, (void *)rq.matrix, rq.size) < 0)
        {
            perror("Can't read size from socket");
            close(sockfd);
            exit(-1);
        }

        if (pthread_mutex_lock(&mutexQueueTask) != 0)
        {
            perror("Can't lock mutex (main process)");
        }
        queueTask.push_back(rq);
        if (pthread_mutex_unlock(&mutexQueueTask) != 0)
        {
            perror("Can't unlock mutex (main process)");
        }
    }

    return 1;
}
