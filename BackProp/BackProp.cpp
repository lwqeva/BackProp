#include <iostream>
#include <memory>
#include <cmath>
#include <cassert>
#include "BackProp.h"

namespace BackProp
{

const float NNet::lr = 0.1;

NNet::NNet(uint n_layers, uint *n_neurons, NodeType *types)
{
	this->n_layers = n_layers;
	this->n_neurons = (uint*) malloc(sizeof(uint)*n_layers);
	node_type = (NodeType*) malloc(sizeof(NodeType)*n_layers);
	if( types )
		memcpy( node_type, types, sizeof(NodeType)*n_layers );
	else
	{
		// Default activation function is Sigmoid for hidden layer 
		// and Linear for the final layer.
		for(uint i = n_layers-2; i > 0 ; --i)
			node_type[i] = Sigmoid;
		node_type[n_layers-1] = Linear;
	}

	W = (float**) malloc(sizeof(float*)*n_layers);
	
	this->n_neurons[0] = n_neurons[0];
	for(uint i = 1; i < n_layers; ++i)
	{
		this->n_neurons[i] = n_neurons[i];
		W[i] = (float*) malloc(sizeof(float)*(n_neurons[i-1]+1)*n_neurons[i]);
		uint id = 0;
		for(uint j = 0; j < n_neurons[i]; ++j)
			for(uint k = 0; k <= n_neurons[i-1]; ++k)
				W[i][id++] = 0.1*(((float)rand()/(float)RAND_MAX) - 0.5);
	}
}

NNet::~NNet()
{}

void NNet::train(DataSet &ds)
{
	assert(ds.dim == n_neurons[0]);
	/// Iterate n epochs
	for(uint epoch = 0; epoch < 500; ++epoch)
	{
		/// Iterate every sample
		for(uint i = 0; i < ds.size; ++i)
		{
		// Forward propagate to get nodes output and error.
			float **signal = allocate_tape();
			memcpy(signal[0], ds.X+i*ds.dim, sizeof(float)*n_neurons[0]);
			for(uint j = 1; j < n_layers; ++j)
				forward_prop(W[j], signal[j-1], n_neurons[j-1],
					signal[j], n_neurons[j], node_type[j]);

//			show_nodes(signal);
		// Backward propagate to get derivatives.
			float **delta = allocate_tape();
			delta[n_layers-1][0] = -2*(ds.Y[i] - signal[n_layers-1][0]);
//		std::cout<< "Error: " << delta[n_layers-1][0] << "\n";
			for(uint j = n_layers-1; j > 1; --j)
				backward_prop(W[j],delta[j-1],n_neurons[j-1],
					signal[j],delta[j],n_neurons[j],node_type[j]);
			
//			show_nodes(delta);
		// Update weights by gradient descent.
			for(uint j = 1; j < n_layers; ++j)
				update_weights(W[j], signal[j-1], n_neurons[j-1], delta[j], n_neurons[j]);
			
		//	show_weights();
			free_tape(signal);
			free_tape(delta);
		}
//		std::cout<<"\n ---- Epoch " << epoch << " End ----\n\n\n";
	}
}

void NNet::predict(DataSet &ds, float *p)
{
	assert(ds.dim == n_neurons[0]);

	for(uint i = 0; i < ds.size; ++i)
	{
		float **signal = allocate_tape();
		memcpy(signal[0], ds.X+i*ds.dim, sizeof(float)*n_neurons[0]);
		for(uint j = 1; j < n_layers; ++j)
				forward_prop(W[j], signal[j-1], n_neurons[j-1], 
					signal[j], n_neurons[j], node_type[j]);

		p[i] = signal[n_layers-1][0];
		free_tape(signal);
	}
}

void NNet::show_weights()
{
	std::cout << " - - - - Weights - - - -\n";
	for(uint i = 1; i < n_layers; ++i)
	{
		std::cout<< "Layer "<< i << " #nodes = " << n_neurons[i] << "\n";
		uint id = 0;
		for(uint j = 0; j < n_neurons[i]; ++j)
		{
			for(uint k = 0; k <= n_neurons[i-1]; ++k)
				std::cout << W[i][id++] << ", ";
			std::cout<< '\n';
		}
		std::cout<< "\n";
	}
}

void NNet::show_nodes(float **s)
{
	std::cout << "- - - - Nodes Status - - - -\n";
	for(uint i = 1; i < n_layers; ++i)
	{
		std::cout<< "Layer "<< i << " #nodes = " << n_neurons[i] << "\n";
		uint id = 0;
		for(uint j = 0; j < n_neurons[i]; ++j)
			std::cout << s[i][j] << ", ";
		std::cout<< "\n\n";
	}
}

void NNet::forward_prop(float *w, float* pre, uint N, float* curr, uint M, NodeType type)
{
	int b = 0;
	for(uint i = 0; i < M; ++i)
	{
		curr[i] = w[b];	// bias
		for(uint j = 0; j < N; ++j)
			curr[i] += pre[j]*w[b+j+1];
		b += (N+1);		// move offset to next neuron's weight

		// Apply activation function
		switch (type)
		{
		case Sigmoid:
			curr[i] = sigmoid(curr[i]);
			break;
		case Linear:
		default:
			break;
		}
		
	}
}

void NNet::backward_prop(float *w, float* pred, uint N, 
						 float* curr, float* currd, uint M, NodeType type)
{
	memset(pred,0,sizeof(float)*N);
	for(uint i = 0; i < M; ++i)
	{
		float deri = currd[i];
		switch (type)
		{
		case Sigmoid:
			deri *= curr[i] * (1-curr[i]);
			break;
		case Linear:
		default:
			break;
		}
		uint b = i + 1;
		for(uint j = 0; j < N; ++j, b+=(N+1))
			pred[j] += deri * w[b];
	}
}

void NNet::update_weights(float *w, float *sig, uint N, float *del, uint M)
{
	uint b = 0;
	for(uint i = 0; i < M; ++i, b+=(1+N) )
	{
		w[b] -= lr * del[i];
		for(uint j = 0; j < N; ++j)
			w[b+j+1] -= lr * sig[j] *del[i];
	}
}

float **NNet::allocate_tape()
{
	float **s;
	s = (float**)malloc(sizeof(float*)*n_layers);
	for(uint i = 0; i < n_layers; ++i)
		s[i] = (float*) malloc(sizeof(float)*n_neurons[i]);
	return s;
}


void NNet::free_tape(float **s)
{
	for(uint i = 0; i < n_layers; ++i)
		free(s[i]);
	free(s);
}


float sigmoid(float x)
{
	return 1.0/(1+exp(-x));
}


void test_initial_weight()
{
	uint sz[] = {1,10,1};
	NNet nnet(3,sz);		// initialize nnet with default architecture

	std::cout << " - - - - Initial Weights - - - -\n";
	nnet.show_weights();

	DataManager dm;		
	DataSet ds = dm.LaodData();	 // Use built-in testing data.
	float *p = (float*) malloc(sizeof(float)*ds.size);

	std::cout<< "\n\n - - - - Data and Initial Output - - - - - \n\n";
	nnet.predict(ds, p);
	for(uint i = 0; i < ds.size; ++i)
	{
		std::cout<< ds.X[i] << "\t" << ds.Y[i] << '\t'
			<< p[i] << "\n";
	}


	std::cout<< "\n\n - - - - Output after Training - - - - - \n\n";
	nnet.train(ds);
	nnet.predict(ds, p);
	for(uint i = 0; i < ds.size; ++i)
	{
		std::cout<< ds.X[i] << "\t" << ds.Y[i] << '\t'
			<< p[i] << "\n";
	}
}

}