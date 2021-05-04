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
CFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -pedantic -Wno-vla-extension $(INCLUDE_PATH) -I./include -I./$(SRCDIR) -DNDEBUG
LFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -Werror -pedantic $(LIBRARY_PATH) -DNDEBUG
LIBS = $(OPENGL_LIBS)

OBJS = $(OBJDIR)/main.o

default: $(TARGET)

all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $(LFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.cpp -o $(OBJDIR)/main.o

clean:
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -f $(TARGET).exe
