# define base OS environment
FROM alpine:latest

# install base dependencies for compiling and testing fdaPDE

# base system dependencies
RUN apk --no-cache add --update \
    g++ \
    libstdc++ \
    cmake \
    git \
    make

# c++ library dependencies
RUN apk add --update \
    eigen-dev

# test dependencies
RUN apk add --update \
    gtest-dev
