#######################################################################################################

# # Mac OS X
#INCLUDE_PATH      = -I/usr/local/include/ -I/usr/local/include/eigen3/
#LIBRARY_PATH      = -L/usr/local/lib/
#OPENGL_LIBS       = -framework OpenGL -framework GLUT

# Linux
INCLUDE_PATH      =
LIBRARY_PATH      =
OPENGL_LIBS       = -lglut -lGL -lX11

# Windows / Cygwin
#INCLUDE_PATH      = -I"C:\MinGW\freeglut\include"
#LIBRARY_PATH      = -L"C:\MinGW\freeglut\lib"
#OPENGL_LIBS       = -lfreeglut -lglew32 -lopengl32

#######################################################################################################

TARGET = mpm
CC = g++
LD = g++
OBJDIR = obj
SRCDIR = src

CFLAGS = -std=c++0x -pedantic -Wno-deprecated -Wall -Wextra -O3 -DNDEBUG
CFLAGS += $(INCLUDE_PATH) -I./include -I./$(SRCDIR)
OMP = -fopenmp
LFLAGS = -std=c++0x -O3 -Wall -Wno-deprecated -Werror -pedantic $(LIBRARY_PATH) -DNDEBUG
LIBS = $(OPENGL_LIBS)

OBJS = $(OBJDIR)/main.o $(OBJDIR)/helper.o

default: $(TARGET)

all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $(OMP) $(LFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp
	mkdir -p $(OBJDIR)
	$(CC) $(OMP) $(CFLAGS) -c $(SRCDIR)/main.cpp -o $(OBJDIR)/main.o

$(OBJDIR)/helper.o: $(SRCDIR)/helper.cpp
	mkdir -p $(OBJDIR)
	$(CC) $(OMP) $(CFLAGS) -c $(SRCDIR)/helper.cpp -o $(OBJDIR)/helper.o

clean:
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -f $(TARGET).exe

