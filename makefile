# final Project
EXE = final

# Root paths
SRC = src/
OUT = bin/

# Src sub paths
ENGINE = $(SRC)Engine/
UTIL = $(ENGINE)Utility/

# Out sub paths
OBJ = $(OUT)obj/

# Include directories
VPATH = src : src/Engine : src/Engine/Objects : src/Utility : src/Vendor/glm/glm : src/Vendor/glm/glm/gtc : src/Vendor/stb_image

# Source files
SRC =	Renderer.o			\
		Sphere.o			\
		Cube.o				\
		SkySphere.o			\
		random.o			\
		stb_image.o			\

# Main target
all: $(EXE)

#  Msys/MinGW
ifeq "$(OS)" "Windows_NT"
CFLG=-O3 -DUSEGLEW
LIBS=-lSDL2main -lSDL2 -lglew32 -lglu32 -lopengl32
CLEAN=rm -f *.exe *.o *.a *.lib *.exp
else
#  OSX
ifeq "$(shell uname)" "Darwin"
RES=$(shell uname -r|sed -E 's/(.).*/\1/'|tr 12 21)
CFLG=-O3 -Wall -g -Wno-deprecated-declarations -DRES=$(RES) -DUSEGLEW -I/opt/homebrew/include -L/opt/homebrew/lib -Qunused-arguments
LIBS=-lglew -lSDL2main -lSDL2 -framework Cocoa -framework OpenGL
#  Linux/Unix/Solaris
else
CFLG=-O3 -Xcompiler -Wall
LIBS=-lSDL2 -lGLU -lGL -lm
endif
#  OSX/Linux/Unix/Solaris
CLEAN=rm -f $(EXE) *.o *.a *.lib *.exp
endif

#  Create archive
objects.a:$(SRC)
	cd $(OBJ) && ar -rcs $@ $^

# Compile rules
%.o: %.c
	nvcc -c $(CFLG) $< -o $(OBJ)$@
%.o: %.cpp
	nvcc -c $(CFLG) $< -o $(OBJ)$@
%.o: %.cu
	nvcc -c -O3 -dlink  $< -o $(OBJ)$@

#  Link
final: Application.o objects.a
	cd $(OBJ) && nvcc -o ../../$@ $^ $(LIBS)

link:
	cd $(OBJ) && ar -rcs objects.a $(SRC)
	cd $(OBJ) && nvcc -o ../../final Application.o objects.a $(LIBS)

#  Clean
clean:
	cd $(OBJ) && $(CLEAN)
	$(CLEAN)
