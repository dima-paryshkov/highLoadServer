#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>

const int countConnections = 100;
int sockfd[countConnections];
float *array[countConnections];
struct sockaddr_in servaddr[countConnections];

void *procedure(void *curI)
{
    int i = (int)curI;

    int n = rand() % 100;
    int degree = 2 + rand() % 8;

    array[i] = (float *)malloc(n * n * sizeof(float));

    for (int j = 0; i < n * n; j++)
        array[i][j] = (float)((rand() % 1000) / (rand() % 10));

    if (connect(sockfd[i], (struct sockaddr *)&servaddr[i], sizeof(servaddr[i])) < 0)
    {
        perror("Can't connect to server");
        close(sockfd[i]);
        exit(-1);
    }
    
    fprintf(stdout, "Connect №%d, ", i);
    fprintf(stdout, "n = %d. Send: ", n);
    for (int j = 0; i < n * n; j++)
        fprintf(stdout, "%f ", array[i][j]);

    int err;
    if ((err = write(sockfd[i], &n, sizeof(int))) < 0)
    {
        perror("Can't write size");
        close(sockfd[i]);
        exit(-1);
    }

    if ((err = write(sockfd[i], &degree, sizeof(int))) < 0)
    {
        perror("Can't write degree");
        close(sockfd[i]);
        exit(-1);
    }

    if ((err = write(sockfd[i], array[i], n * n * sizeof(float))) < 0)
    {
        perror("Can't write array");
        close(sockfd[i]);
        exit(-1);
    }

    if ((n = read(sockfd[i], array[i], n * n * sizeof(float))) < 0)
    {
        perror("Can\'t read\n");
        close(sockfd[i]);
        exit(-1);
    }
    
    fprintf(stdout, "Connect №%d, ", i);
    fprintf(stdout, "n = %d. Received: ", n);
    for (int j = 0; i < n * n; j++)
        fprintf(stdout, "%f ", array[i][j]);

    if (close(sockfd[i]) != 0)
    {
        perror("Can't close socket");
        exit(-1);
    }
    pthread_exit(NULL);
}

void main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./client <IP address>\n");
        exit(-1);
    }

    for (int i = 0; i < countConnections; i++)
    {
        if ((sockfd[i] = socket(PF_INET, SOCK_STREAM, 0)) < 0)
        {
            perror("Can't get socket");
            exit(-1);
        }
    }

    for (int i = 0; i < countConnections; i++)
    {
        bzero(&servaddr, sizeof(servaddr));
        servaddr[i].sin_family = AF_INET;
        servaddr[i].sin_port = htons(51000);
        if (inet_aton(argv[1], &servaddr[i].sin_addr) == 0)
        {
            fprintf(stderr, "Invalid IP address\n");
            for (int j = 0; j < i; j++)
                close(sockfd[j]);
            exit(-1);
        }
    }

    pthread_t pthread[countConnections];

    for (int i = 0; i < countConnections; i++)
    {
        if (pthread_create(&pthread[i], (pthread_attr_t *)NULL, procedure, (void *)i) != 0)
        {
            perror("Can't create thread");
            exit(-1);
        }
        sleep(5);
    }

    for (int i = 0; i < countConnections; i++)
    {
        if (pthread_join(pthread[i], NULL) != 0)
        {
            perror("Can't join thread");
            exit(-1);
        }
    }
}