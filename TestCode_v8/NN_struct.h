
#ifdef DATA_TYPE_FLOAT 
  #define DATA_TYPE float
  #define EXP_LIMIT 78.0  // limit 88.xx but we need to factor in accumulation for softmax
  #define EXP(a) expl(a)
#else
  #define DATA_TYPE double
  #define EXP_LIMIT 699.0 // limit is 709.xx but we need to factor in accumulation for softmax
  #define EXP(a) exp(a)
#endif

// dummy input for testing
DATA_TYPE input[IN_NUM_CHANNEL][IN_VEC_SIZE];

// dummy output for testing
DATA_TYPE hat_y[OUT_VEC_SIZE];    // target output
DATA_TYPE y[OUT_VEC_SIZE];        // output after forward propagation


// input node or the very first layer and for flatten layer but keep in mind that there can be many channels
typedef struct inVector_t {
	DATA_TYPE X;
} inVector;


typedef struct neuron_t {
	DATA_TYPE * W;
	DATA_TYPE B;
	DATA_TYPE X;

	// For back propagation, convention, dLa means dL/da or partial derivative of Loss over Accumulative output
	DATA_TYPE * dW;
	DATA_TYPE dLa;
	DATA_TYPE dB;

} neuron;

// this contains the contents of a single filter but keep in mind that the size 
// depends on number of input channels also, hence a double pointer for filter!
// but the output value is purely dependent on number of channels in current layer and hence single pointer
typedef struct cnn1D_t {
	DATA_TYPE ** FW;
	DATA_TYPE B;
	DATA_TYPE * FX;
	
	// we need something like dLa for CNN
	DATA_TYPE ** dFW;
	DATA_TYPE dB;
} cnn1D;


// support two types, average and max FOR NOW ITS ONLY MAX POOLING
typedef struct pooling1D_t {
	// uint8_t type;            // placeholder for future expansion to support average pooling
	uint8_t * prevLayerWinIndx;   // useful for backpropagation, we need to store one value for every filter applied. Size is same as outVectorSize
	DATA_TYPE * FX;
} pooling1D;


// there is no way to modify the structure based on layer type, this means that memory wasting is happening here
typedef struct layer_t {
	neuron * p_neuron;        // one pointer index for each neuron in a layer
	cnn1D * p_cnn1D;          // one pointer index for each output channel in a layer
	pooling1D * p_pooling1D;  // one pointer index for each output channel (same number of input channel) in a layer
	inVector * p_inVector;    // will be used for flatten layer, one index for each node in a layer after flattening
	uint8_t type;
	uint8_t numNeuron;
	uint8_t numChannelCurrLayer;
	uint8_t filtVectorSize;
	uint8_t outVectorSize;
} layer;

// to keep track of previous layer info as we build the network
typedef struct prevLayerInfo_t {
	uint8_t layerIndx;
	uint8_t vectorSize;   // used to store both number of neurons for DNN layers and vector size for CNN1D layers
	uint8_t numChannels;
} prevLayerInfo;