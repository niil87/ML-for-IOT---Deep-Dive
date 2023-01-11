#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<time.h>
#include <math.h>
#include<inttypes.h>

#define DATA_TYPE_FLOAT
#define DEBUG 0
#define LEARNING_RATE 0.01
#define EPOCH 10000

#define IN_VEC_SIZE 20
#define OUT_VEC_SIZE 20
#define IN_NUM_CHANNEL 1


#include "NN_struct.h"

layer * p_layers = NULL;
prevLayerInfo * Lprev = NULL;

#include "NN_function.h"

// Code is contributed by Nikhil Challa


// Convention to be following for multi-dimension
// 1. [filter indx or channel indx][vector]


// Code has the following restriction
// 1. stride should filter size for POOLING
// 2. stride is equal to 1 for filters or CNN layers
// 3. We will use default value as much as possible, matching default values in tensorflow
// 4. No support for average pooling, only max pooling


// REMOVE FOR DEPLOYMENT // need to test with multiple channels
void generateData() {
	int indx = rand()*1.0/RAND_MAX*OUT_VEC_SIZE;
	//indx = 1;
	for (int j =0; j < OUT_VEC_SIZE; j++) {
		input[0][j] = 0.0;
		hat_y[j] = 0.0;
	}
	input[0][indx] = 1.0;
	hat_y[indx] = 1.0;
}

int main() {
    printf("Hello World!\n");
	
	// Defaulting the 
	Lprev = (prevLayerInfo*)malloc(sizeof(prevLayerInfo));
	Lprev->numChannels = IN_NUM_CHANNEL;
	Lprev->vectorSize = IN_VEC_SIZE;
	Lprev->layerIndx = 0;

	printf("Allocating memory for first layer to begin with!\n");
	p_layers = (layer*)malloc(1*sizeof(layer));
	printf("p_layers : %d\n",sizeof(p_layers));
	
	printf("Size of layer : %d\n",sizeof(layer));
	printf("Size of size_t : %d\n",sizeof(size_t));
	
	createCNN1dLayer(1,4);
	//createPooling1dLayer(2);
	createFlattenLayer();
	createNeuronLayer(3);
	createNeuronLayer(OUT_VEC_SIZE);
	compileNetwork();
	
	numLayers = Lprev->layerIndx + 1;
	
	printf("Done creating layers with total number: %" PRIu8 "\n",numLayers);
	
	// printf("Layer indx 3, num neuron : %" PRIu8 "\n",p_layers[3].numNeuron);
	
	forwardProp();
	for (uint8_t j = 0; j < OUT_VEC_SIZE; j++) {
		printf("output value for hat_y:%f\n",hat_y[j]);
	}
	for (uint8_t j = 0; j < OUT_VEC_SIZE; j++) {
		printf("output value for y:%f\n",y[j]);
	}
	
/* 	for (int i = 0; i < EPOCH; i++) {
		printf("Running epoch:%d\n",i);
		generateData();
		forwardProp();
		backwardProp();
		
		
		if (i > EPOCH - 10) { 
		
			for (uint8_t j = 0; j < OUT_VEC_SIZE; j++) {
				printf("output value for hat_y:%f\n",hat_y[j]);
			}
			for (uint8_t j = 0; j < OUT_VEC_SIZE; j++) {
				printf("output value for y:%f\n",y[j]);
			}
		}
		
	} */
	
	

	
	free(p_layers);
	
	
	
	return 0;
}


