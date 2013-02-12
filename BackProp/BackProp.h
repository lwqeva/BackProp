#pragma once
#include "JobManager.h"

namespace BackProp
{
	class NNet
	{
	public:
		// Types of activiation function
		enum NodeType { Linear, Sigmoid };

		// NNet constructor
		// Allocate and initialized weights and descriptive variables.
		NNet(uint n_layers, uint *n_neurons, NodeType *types=0);
		// NNet desctroyer.
		// Deallocate space.
		~NNet();

		// Train neural net with given data set.
		void train(DataSet &ds);
		// Make prediction based on input data set.
		void predict(DataSet &ds, float *p);

		// Display weights to cout.
		void show_weights();
	private:

		// Propagate output from previous layer to current layer.
		void forward_prop(float *w, float *pre, uint N, float *curr, uint M, NodeType type);
		// Propagate output from current layer to previous layer
		void backward_prop(float *w, float *pred, uint N, 
			float* curr, float* currd, uint M, NodeType type);
		// Update weights by gradient descent
		void update_weights(float *w, float *sig, uint N, float *del, uint M);
		// Allocate space to store intermediate results.
		float **allocate_tape();
		// Deallocate space of intermediate results
		void free_tape(float **s);

		// Display node status to cout.
		void show_nodes(float **s);

		static const float lr;	// learning rate

		float **W;	// Weights of the neural net.
		NodeType *node_type;	// Activation function type of each layer
		uint n_layers;	// number of layers
		uint *n_neurons;	// number of neurons in each layer.
	};

	inline float sigmoid(float x);
	void test_initial_weight();
}
