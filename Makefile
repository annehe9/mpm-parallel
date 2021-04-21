#######################################################################################################

# # Mac OS X
#INCLUDE_PATH      = -I/usr/local/include/ -I/usr/local/include/eigen3/
#LIBRARY_PATH      = -L/usr/local/lib/
#OPENGL_LIBS       = -framework OpenGL -framework GLUT

# Linux
INCLUDE_PATH      =
LIBRARY_PATH      = -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
OPENGL_LIBS       = -lglut -lGL -lX11

# Windows / Cygwin
#INCLUDE_PATH      = -I"C:\MinGW\freeglut\include"
#LIBRARY_PATH      = -L"C:\MinGW\freeglut\lib"
#OPENGL_LIBS       = -lfreeglut -lglew32 -lopengl32

#######################################################################################################

TARGET = mpm
CC = g++
NVCC = nvcc
LD = g++
OBJDIR = obj
SRCDIR = src
CFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -pedantic -Wno-vla-extension $(INCLUDE_PATH) -I./include -I./$(SRCDIR) -DNDEBUG
LFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -Werror -pedantic $(LIBRARY_PATH) -DNDEBUG
LIBS = $(OPENGL_LIBS)
NVCCFLAGS = -O3 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

OBJS = $(OBJDIR)/main.o $(OBJDIR)/cudaMPM.o

default: $(TARGET)

all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $(LFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.cpp -o $(OBJDIR)/main.o

$(OBJDIR)/cudaMPM.o: $(SRCDIR)/cudaMPM.cu
	mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $(SRCDIR)/cudaMPM.cu -o $(OBJDIR)/cudaMPM.o
clean:
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -f $(TARGET).exe

