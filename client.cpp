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
#include <iostream>
#include <chrono>
#include <iostream>
#include <fstream>

const int countConnections = 200;
int sockfd[countConnections];
int *array[countConnections];
struct sockaddr_in servaddr;

int id[countConnections];

void *procedure(void *curI)
{
    int *i = (int *)curI;
    int n = 128;
    int degree = 20;

    array[*i] = (int *)malloc(n * n * sizeof(int));

    for (int j = 0; j < n * n; j++)
        array[*i][j] = rand() % 1000;

    if (connect(sockfd[*i], (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        perror("Can't connect to server");
        close(sockfd[*i]);
        exit(-1);
    }

    int err;
    if ((err = write(sockfd[*i], &n, sizeof(int))) < 0)
    {
        perror("Can't write size");
        close(sockfd[*i]);
        exit(-1);
    }

    if ((err = write(sockfd[*i], &degree, sizeof(int))) < 0)
    {
        perror("Can't write degree");
        close(sockfd[*i]);
        exit(-1);
    }

    if ((err = write(sockfd[*i], (void *)array[*i], n * n * sizeof(int))) < 0)
    {
        perror("Can't write array");
        close(sockfd[*i]);
        exit(-1);
    }

    // fprintf(stdout, "Connect %d, n = %d, count of send bytes = %d\n", *i, n, err);

    if ((err = read(sockfd[*i], &array[*i][0], n * n * sizeof(int))) < 0)
    {
        perror("Can\'t read\n");
        close(sockfd[*i]);
        exit(-1);
    }

    // fprintf(stdout, "Connect %d, n = %d, count of recv bytes = %d\n", *i, n, err);
 
    if (close(sockfd[*i]) != 0)
    {
        perror("Can't close socket");
        exit(-1);
    }
    pthread_exit(NULL);
}

int main(int argc, char **argv)
{
    auto begin = std::chrono::steady_clock::now();
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
        servaddr.sin_family = AF_INET;
        servaddr.sin_port = htons(51000);
        if (inet_aton(argv[1], &servaddr.sin_addr) == 0)
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
        id[i] = i;
        if (pthread_create(&pthread[i], (pthread_attr_t *)NULL, procedure, (void *)&id[i]) != 0)
        {
            perror("Can't create thread");
            exit(-1);
        }
    }

    for (int i = 0; i < countConnections; i++)
    {
        if (pthread_join(pthread[i], NULL) != 0)
        {
            perror("Can't join thread");
            exit(-1);
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Time: " << time.count() <<  std::endl;

    return 1;
}