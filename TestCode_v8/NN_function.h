#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define fRAND ( rand()*1.0/RAND_MAX-0.5 )*2   // random number generator between -1 and +1 
#define ACT(a) MAX(a,0)    // RELU(a)

#define INPUT 0
#define NEURON 1
#define CNN_1D 2
#define CNN_2D 3
#define POOL_1D 4
#define POOL_2D 5
#define FLATTEN 6
#define DROPOUT 7
#define NEURON_SOFTMAX 8

/* // placeholder for future expansion
#define MAX_POOL 0
#define AVG_POOL 1 */

uint8_t numLayers;

// creates a neuron in a layer
void createNeuron(int numNeuronPrevLayer, neuron * N) {

	N->B = fRAND;
	#if DEBUG
		printf("Value of Bias value:%f\n", N->B);
	#endif
	N->W = (DATA_TYPE * )malloc(numNeuronPrevLayer*sizeof(DATA_TYPE));
	N->dW = (DATA_TYPE * )malloc(numNeuronPrevLayer*sizeof(DATA_TYPE));
	// initializing values of W to rand and dW to 0
	for (uint8_t i = 0; i < numNeuronPrevLayer; i++) {
		N->W[i] = fRAND;
		N->dW[i] = 0.0;
		#if DEBUG
			printf("Value of weight indx:%" PRIu8 ", value:%f\n", i, N->W[i]);
		#endif
	}
	N->dLa = 0.0;
	N->dB = 0.0;

}

// creates a 1D Conv filter in a layer
void createCnn1D (uint8_t numChannelPrevLayer, uint8_t filtVectorSize, uint8_t outVectorSize, cnn1D * C) {

	//printf("initiating memory for filter with num of channels in prev layer as: %" PRIu8 "\n",numChannelPrevLayer);
	C->fW = (DATA_TYPE ** )malloc( numChannelPrevLayer * sizeof(DATA_TYPE *) );
	C->dfW = (DATA_TYPE ** )malloc( numChannelPrevLayer * sizeof(DATA_TYPE *) );
 	for (uint8_t i = 0; i < numChannelPrevLayer; i++) {
		C->fW[i] = (DATA_TYPE *)malloc(filtVectorSize*sizeof(DATA_TYPE));
		C->dfW[i] = (DATA_TYPE *)malloc(filtVectorSize*sizeof(DATA_TYPE));
	}
	
	// initializing all values to random
	for (uint8_t i = 0; i < numChannelPrevLayer; i++) {
		for (uint8_t j = 0; j < filtVectorSize; j++) {
			C->fW[i][j] = fRAND;
			C->dfW[i][j] = 0.0;
		}
	}
	
	C->B = fRAND;
	C->dB = 0.0;	
		
	C->fX = (DATA_TYPE *)malloc(outVectorSize*sizeof(DATA_TYPE ));
	// initializing all values to random
	for (uint8_t j = 0; j < outVectorSize; j++) {
		C->fX[j] = 0.0;
	}

}


// creates a 1D Conv filter in a layer
void createPool1D (uint8_t outVectorSize, pooling1D * P) {
	
	P->fX = (DATA_TYPE * )malloc(outVectorSize*sizeof(DATA_TYPE));
	// initializing all values to random
	for (uint8_t i = 0; i < outVectorSize; i++) {
		P->fX[i] = 0.0;
	}
	P->prevLayerWinIndx = 0;

}


void compileNetwork() {
	
	if (p_layers[Lprev->layerIndx - 1].type == NEURON) {
		p_layers[Lprev->layerIndx - 1].type = NEURON_SOFTMAX;
	} else {
		perror("Seriously you need neuron layer before compiling network... \n");
		exit(0);
	}
	// this is last layer... no additional memory and no memory allocation required as we directly use output vector y

	
	// resizing back by 1 layer size
	Lprev->layerIndx = Lprev->layerIndx - 1;
	p_layers = (layer *)realloc(p_layers,(Lprev->layerIndx + 1)*sizeof(layer));

	
}

void createFlattenLayer() {

	// updating for next layer   //////////////////////
	//printf("Type of previous layer : %" PRIu8 "\n",p_layers[Lprev->layerIndx - 1].type);
	uint8_t outputVectorSize;

	// no support for 2D input, works for previous layeing being CNN_1D, POOL_1D, or direct input feed
	outputVectorSize = Lprev->vectorSize * Lprev->numChannels;

	p_layers[Lprev->layerIndx].p_inVector = (inVector * )malloc(outputVectorSize*sizeof(inVector));
	// setting pointers to null for better error checks
	p_layers[Lprev->layerIndx].p_cnn1D = NULL;
	p_layers[Lprev->layerIndx].p_pooling1D = NULL;
	p_layers[Lprev->layerIndx].p_neuron = NULL;
	
	
	for (uint8_t i = 0; i < Lprev->vectorSize; i++) {
		p_layers[Lprev->layerIndx].p_inVector[i].X = 0.0;
	}
	
	p_layers[Lprev->layerIndx].type = FLATTEN;
	p_layers[Lprev->layerIndx].numNeuron = outputVectorSize;
	p_layers[Lprev->layerIndx].numChannelCurrLayer = 0;
	p_layers[Lprev->layerIndx].filtVectorSize = 0;
	p_layers[Lprev->layerIndx].outVectorSize = 0;
	
	printf("New vector size after FLATTEN Call : %" PRIu8 "\n",outputVectorSize);
	
	// updating for next layer   //////////////////////
	Lprev->vectorSize = outputVectorSize;
	Lprev->numChannels = 0;
	Lprev->layerIndx += 1;
	p_layers = (layer *)realloc(p_layers,(Lprev->layerIndx + 1)*sizeof(layer));
	//////////////////////////////////////////////////
}

// creates a 1D pooling layer
void createPooling1dLayer (uint8_t filtVectorSize) {

	printf("Creating POOLING layer with filter size: %" PRIu8 "\n",filtVectorSize);
	
	uint8_t numChannelCurrLayer = Lprev->numChannels;
	
	p_layers[Lprev->layerIndx].p_pooling1D = (pooling1D*)malloc(numChannelCurrLayer*sizeof(pooling1D));
	// setting pointers to null for better error checks
	p_layers[Lprev->layerIndx].p_inVector = NULL;
	p_layers[Lprev->layerIndx].p_cnn1D = NULL;
	p_layers[Lprev->layerIndx].p_neuron = NULL;
	
	uint8_t outputVectorSize = Lprev->vectorSize / filtVectorSize + Lprev->vectorSize % filtVectorSize;
	for (uint8_t i = 0; i < numChannelCurrLayer; i++) {
		createPool1D(outputVectorSize,(p_layers[Lprev->layerIndx].p_pooling1D + i));
	}
	
	p_layers[Lprev->layerIndx].type = POOL_1D;
	p_layers[Lprev->layerIndx].numNeuron = 0;
	p_layers[Lprev->layerIndx].numChannelCurrLayer = Lprev->numChannels;
	p_layers[Lprev->layerIndx].filtVectorSize = filtVectorSize;
	p_layers[Lprev->layerIndx].outVectorSize = outputVectorSize;
	
	// updating for next layer   //////////////////////
	Lprev->numChannels = Lprev->numChannels;
	Lprev->vectorSize = outputVectorSize;
	Lprev->layerIndx += 1;
	p_layers = (layer *)realloc(p_layers,(Lprev->layerIndx + 1)*sizeof(layer));
	//////////////////////////////////////////////////
	
	printf("New vector size after POOL 1D Call : %" PRIu8 "\n",Lprev->vectorSize);
}


void createNeuronLayer (uint8_t numNeuron) {

	printf("Creating neuron layer of size %" PRIu8 "\n",numNeuron);

	p_layers[Lprev->layerIndx].p_neuron = (neuron*)malloc(numNeuron*sizeof(neuron));
	// setting pointers to null for better error checks
	p_layers[Lprev->layerIndx].p_inVector = NULL;
	p_layers[Lprev->layerIndx].p_cnn1D = NULL;
	p_layers[Lprev->layerIndx].p_pooling1D = NULL;

	if (Lprev->layerIndx != 0) {
		for (int i = 0; i < numNeuron; i++) {
			createNeuron(Lprev->vectorSize, (p_layers[Lprev->layerIndx].p_neuron + i));
		}
	}

	p_layers[Lprev->layerIndx].type = NEURON;
	p_layers[Lprev->layerIndx].numNeuron = numNeuron;
	p_layers[Lprev->layerIndx].numChannelCurrLayer = 0;
	p_layers[Lprev->layerIndx].filtVectorSize = 0;
	p_layers[Lprev->layerIndx].outVectorSize = 0;

	// updating for next layer   //////////////////////
	Lprev->numChannels = 0;
	Lprev->vectorSize = numNeuron;
	Lprev->layerIndx += 1;
	p_layers = (layer *)realloc(p_layers,(Lprev->layerIndx + 1)*sizeof(layer));
	//////////////////////////////////////////////////
}

void createCNN1dLayer (uint8_t numChannelCurrLayer, uint8_t filtVectorSize) {
	
	printf("Creating CNN1D layer with number of filters: %" PRIu8 ", and filter size %" PRIu8 "\n",numChannelCurrLayer, filtVectorSize);
	
	p_layers[Lprev->layerIndx].p_cnn1D = (cnn1D*)malloc(numChannelCurrLayer*sizeof(cnn1D));
	// setting pointers to null for better error checks
	p_layers[Lprev->layerIndx].p_neuron = NULL;
	p_layers[Lprev->layerIndx].p_inVector = NULL;
	p_layers[Lprev->layerIndx].p_pooling1D = NULL;
	
	uint8_t outputVectorSize = Lprev->vectorSize / filtVectorSize + Lprev->vectorSize % filtVectorSize;
	printf("Output vector size for CNN layer: %" PRIu8 "\n",outputVectorSize);
	
	for (uint8_t i = 0; i < numChannelCurrLayer; i++) {
		createCnn1D(Lprev->numChannels,filtVectorSize,outputVectorSize,(p_layers[Lprev->layerIndx].p_cnn1D + i));
	}
	
	p_layers[Lprev->layerIndx].type = CNN_1D;
	p_layers[Lprev->layerIndx].numNeuron = 0;
	p_layers[Lprev->layerIndx].numChannelCurrLayer = numChannelCurrLayer;
	p_layers[Lprev->layerIndx].filtVectorSize = filtVectorSize;
	p_layers[Lprev->layerIndx].outVectorSize = outputVectorSize;
	
	// updating for next layer   //////////////////////
	Lprev->numChannels = numChannelCurrLayer;
	Lprev->vectorSize = outputVectorSize;
	Lprev->layerIndx += 1;
	p_layers = (layer *)realloc(p_layers,(Lprev->layerIndx + 1)*sizeof(layer));
	//////////////////////////////////////////////////
	
	printf("New vector size after CNN 1D Call : %" PRIu8 "\n",Lprev->vectorSize);
}


// Equation (8)
DATA_TYPE AccFunction (int8_t layerIndx, int8_t nodeIndx, int8_t prevLayerType) {
	DATA_TYPE A = 0;

	for (int8_t k = 0; k < p_layers[layerIndx - 1].numNeuron; k++) {

		// updating weights/bais and resetting gradient value if non-zero
		if (p_layers[layerIndx].p_neuron[nodeIndx].dW[k] != 0.0 ) {
			p_layers[layerIndx].p_neuron[nodeIndx].W[k] += p_layers[layerIndx].p_neuron[nodeIndx].dW[k];
			p_layers[layerIndx].p_neuron[nodeIndx].dW[k] = 0.0;
		}
		if (prevLayerType == FLATTEN) {
			A += p_layers[layerIndx].p_neuron[nodeIndx].W[k] * p_layers[layerIndx - 1].p_inVector[k].X;
		} else {
			A += p_layers[layerIndx].p_neuron[nodeIndx].W[k] * p_layers[layerIndx - 1].p_neuron[k].X;
		}

	}

	if (p_layers[layerIndx].p_neuron[nodeIndx].dB != 0.0 ) {
		p_layers[layerIndx].p_neuron[nodeIndx].B += p_layers[layerIndx].p_neuron[nodeIndx].dB;
		p_layers[layerIndx].p_neuron[nodeIndx].dB = 0.0;
	}
	A += p_layers[layerIndx].p_neuron[nodeIndx].B;

	return A;
}


// this function is to calculate dLa
DATA_TYPE dLossCalc( int layerIndx, int nodeIndx) {

	DATA_TYPE Sum = 0;
	// for the last layer, we use complex computation
	if (layerIndx == numLayers - 1) {	
		Sum = y[nodeIndx] - hat_y[nodeIndx];										// Equation (17)
	// for all except last layer, we use simple aggregate of dLa
	} else if (AccFunction(layerIndx, nodeIndx,p_layers[layerIndx-1].type) > 0)  {   							
		for (int i = 0; i < p_layers[layerIndx + 1].numNeuron; i++) {
			Sum += p_layers[layerIndx + 1].p_neuron[i].dLa * p_layers[layerIndx + 1].p_neuron[i].W[nodeIndx]; 	// Equation (24)
		}
	} else {   																		// refer to "Neat Trick" and Equation (21)
		Sum = 0;
	}

	return Sum;
}

void forwardProp() {
	
	DATA_TYPE Fsum = 0;
	uint8_t maxIndx = 0;
	// Propagating through network
	for (uint8_t i = 0; i < numLayers; i++) {
		// printf("Processing through layer indx:%" PRIu8 "\n",i);
		if (p_layers[i].type == NEURON) {
			// for first layer, we just feed in input
			if ( i == 0 ) {
				for (uint8_t j = 0; j < p_layers[i].numNeuron;j++) {
					p_layers[i].p_neuron[j].X = input[0][j];
				}
			// for subsequent layers, we need to perform RELU
			} else {
				for (uint8_t j = 0; j < p_layers[i].numNeuron;j++) {
					p_layers[i].p_neuron[j].X = ACT(AccFunction(i,j,p_layers[i-1].type));				// Equation (21)	
					#if DEBUG
						printf("output X:%f\n",p_layers[i].p_neuron[j].X);
					#endif
				}
			}
		} else if (p_layers[i].type == NEURON_SOFTMAX) { 
		  // softmax functionality but require normalizing performed later
			for (uint8_t j = 0; j < p_layers[i].numNeuron;j++) {
				y[j] = AccFunction(i,j,p_layers[i-1].type);
				// tracking the max index
				if ( ( j > 0 ) && (abs(y[maxIndx]) < abs(y[j])) ) {
					maxIndx = j;
				}
			}
		} else if (p_layers[i].type == CNN_1D) {
			
			for (uint8_t jnew = 0; jnew < p_layers[i].numChannelCurrLayer; jnew++ ) {
				if (i == 0) {
					for (uint8_t j = 0;j < IN_NUM_CHANNEL; j++ ) {
						for (uint8_t k = 0; k < p_layers[i].outVectorSize; k++) {
							DATA_TYPE temp = 0;
							uint8_t m;
							for (m = 0 + k*p_layers[i].filtVectorSize; m < (k+1)*p_layers[i].filtVectorSize; m++) {
								temp += input[j][m] * p_layers[i].p_cnn1D[jnew].fW[j][m - k*p_layers[i].filtVectorSize];
							}
							p_layers[i].p_cnn1D[jnew].fX[k] = temp + p_layers[i].p_cnn1D[jnew].B;
						}
					}
				} else if (p_layers[i-1].type == CNN_1D) {
					for (uint8_t j = 0;j < p_layers[i-1].numChannelCurrLayer; j++ ) {
						for (uint8_t k = 0; k < p_layers[i].outVectorSize; k++) {
							DATA_TYPE temp = 0;
							uint8_t m;
							for (m = 0 + k*p_layers[i].filtVectorSize; m < (k+1)*p_layers[i].filtVectorSize; m++) {
								temp += p_layers[i-1].p_cnn1D[j].fX[m] * p_layers[i].p_cnn1D[jnew].fW[j][m - k*p_layers[i].filtVectorSize];
							}
							p_layers[i].p_cnn1D[jnew].fX[k] = temp + p_layers[i].p_cnn1D[jnew].B;
						}
					}
				} else if (p_layers[i-1].type == POOL_1D) {
					for (uint8_t j = 0;j < p_layers[i-1].numChannelCurrLayer; j++ ) {
						for (uint8_t k = 0; k < p_layers[i].outVectorSize; k++) {
							DATA_TYPE temp = 0;
							uint8_t m;
							for (m = 0 + k*p_layers[i].filtVectorSize; m < (k+1)*p_layers[i].filtVectorSize; m++) {
								temp += p_layers[i-1].p_pooling1D[j].fX[m] * p_layers[i].p_cnn1D[jnew].fW[j][m - k*p_layers[i].filtVectorSize];
							}
							p_layers[i].p_cnn1D[jnew].fX[k] = temp + p_layers[i].p_cnn1D[jnew].B;
						}
					}	
				} else {
					//Need to build for CNN_2D and POOL_2D
					perror("Sorry no support for CNN_2D or POOL_2D... \n");
					exit(0);
				}
			}
		} else if (p_layers[i].type == POOL_1D) {
			// only max pooling
			for (uint8_t j = 0;j < p_layers[i].numChannelCurrLayer; j++ ) {
				for (uint8_t k = 0; k < p_layers[i].outVectorSize; k++) {
					DATA_TYPE temp = 0;
					uint8_t m;
					if (i == 0) {
						for (m = 0 + k*p_layers[i].filtVectorSize; m < (k+1)*p_layers[i].filtVectorSize; m++) {
							if (temp <  input[j][m]) {
								temp = input[j][m];
							}
						}
					} else {
						for (m = 0 + k*p_layers[i].filtVectorSize; m < (k+1)*p_layers[i].filtVectorSize; m++) {
							if (temp <  p_layers[i-1].p_cnn1D[j].fX[m]) {
								temp = p_layers[i-1].p_cnn1D[j].fX[m];
							}
						}
					}
					p_layers[i].p_pooling1D[j].fX[k] = temp;
					p_layers[i].p_pooling1D[j].prevLayerWinIndx = m - k*p_layers[i].filtVectorSize;
				}
			}	
			
		} else if (p_layers[i].type == FLATTEN) {
			// to flatten, we start with first channel, cover all elements in first row and then move to next row
			uint8_t count = 0;
			if (i == 0) {
				for (uint8_t j = 0; j < IN_NUM_CHANNEL;j++) {
					for (uint8_t k = 0; k < IN_VEC_SIZE;k++) {
						p_layers[i].p_inVector[count].X = input[j][k];
						count += 1;
					}
				}
			} else {
				for (uint8_t j = 0; j < p_layers[i-1].numChannelCurrLayer;j++) {
					for (uint8_t k = 0; k < p_layers[i-1].outVectorSize;k++) {
						if (p_layers[i-1].type == POOL_1D) {
							p_layers[i].p_inVector[count].X = p_layers[i-1].p_pooling1D[j].fX[k];
						} else {
							p_layers[i].p_inVector[count].X = p_layers[i-1].p_cnn1D[j].fX[k];
						}
						count += 1;
					}
				}
				printf("After flatten verifying total count:%" PRIu8 "\n",count);
			}
			
		} else {
			//Need to build for CNN_2D and POOL_2D
			perror("Sorry no support for CNN_2D or POOL_2D... \n");
			exit(0);
		}
	}

  // performing exp but ensuring we dont exceed 709 or 88 in any terms 
	DATA_TYPE norm = abs(y[maxIndx]);
	if (norm > EXP_LIMIT) {
#if DEBUG
		printf("Max limit exceeded for exp:");
		printf("%f\n",norm);
#endif
		norm = norm / EXP_LIMIT;
#if DEBUG
		printf("New divising factor:");
		printf("%f\n",norm);
#endif
	} else {
		norm = 1.0;
	}
	for (uint8_t j = 0; j < OUT_VEC_SIZE;j++) {
		uint8_t flag = 0;
		y[j] = EXP(y[j]/norm);
		Fsum += y[j];
	}

  // final normalizing for softmax
	for (uint8_t j = 0; j < OUT_VEC_SIZE;j++) {
		y[j] = y[j]/Fsum;
	}
}

void backwardProp() {
	for (uint8_t i = numLayers - 1; i > 0; i--) {
		#if DEBUG
			printf("Back prop : Layer index:%" PRIu8 "\n",i);
		#endif
    // tracing each node in the layer.
		for (uint8_t j = 0; j < p_layers[i].numNeuron; j++) {
			// first checking if drivative of activation function is 0 or not! NEED TO UPGRADE TO ALLOW ACTIVATION FUNCTION OTHER THAN RELU
			p_layers[i].p_neuron[j].dLa = dLossCalc(i, j);
			
			#if DEBUG
				printf("Value of dLa:%f\n",p_layers[i].p_neuron[j].dLa);
			#endif

			for (uint8_t k = 0; k < p_layers[i-1].numNeuron; k++) {
				p_layers[i].p_neuron[j].dW[k] = -LEARNING_RATE * p_layers[i].p_neuron[j].dLa * p_layers[i - 1].p_neuron[k].X;
				#if DEBUG
					printf("Value of dW:%f\n",p_layers[i].p_neuron[j].dW[k]);
				#endif
			}
			p_layers[i].p_neuron[j].dB = -LEARNING_RATE * p_layers[i].p_neuron[j].dLa;

		}
	}
}
