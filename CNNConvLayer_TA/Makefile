CXX = nvcc
TARGET = CNNConvLayer

all: CNNConvLayer_TA.cu
	$(CXX) $< -o $(TARGET)

.PHONY: clean run

clean:
	rm -f $(TARGET) 

run:
	./$(TARGET)
	
	