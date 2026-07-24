# Nom de l'exécutable
TARGET = dynamo

# Compilateur
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
# Linker libraries (older GCC may require -lstdc++fs for std::filesystem)
LIBS = -lstdc++fs

# Fichiers sources et objets
SRC = main.cpp  dynamo.cpp
OBJ = $(SRC:.cpp=.o)

# Règle par défaut
all: $(TARGET)

# Lien de l'exécutable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Compilation des .cpp en .o
%.o: %.cpp dynamo.h configfile.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

#Nettoyage
clean:
	rm -f $(OBJ) $(TARGET)
ifneq ($(filter-out $@,$(MAKECMDGOALS)),)
	for d in $(filter-out $@,$(MAKECMDGOALS)); do \
		if [ -d "$$d" ]; then \
			rm -rf "$$d"/* "$$d"/.[!.]* "$$d"/..?* 2>/dev/null || true; \
		else \
			rm -rf "$$d" 2>/dev/null || true; \
		fi; \
	done
endif

# simu
run: $(TARGET)
	./$(TARGET).exe $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

