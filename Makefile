CXX = nvcc -arch=sm_35 -rdc=true 
TARGET = CNNConvLayer

all: CNNConvLayer.cu
	$(CXX) $< -o $(TARGET)

.PHONY: clean run

clean:
	rm -f $(TARGET) 

run:
	./$(TARGET)
	
	
