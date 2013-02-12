#include <memory>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include "JobManager.h"
using namespace std;


/* Implementation of class JobManager  /// /// ///
*/
JobManager::JobManager()
{
	job_in_queue = 0;
}

JobManager* JobManager::jobmgr = 0;
JobManager& JobManager::GetManager()
{
	if( !jobmgr )
		jobmgr = new JobManager();
	return *jobmgr;
}

JobManager::Job::Job(void (*func)(void))
{
	this->func = func;
}

void JobManager::Job::run()
{
	(*func)();
}

bool JobManager::AppendJob(Job *job)
{
	if( job_in_queue == MAX_JOBS)
		return false;
	jobs[job_in_queue++] = job;
	return true;
}

void JobManager::Launch()
{
	for(int i = 0; i < job_in_queue; ++i)
		jobs[i]->run();
	std::cout<< "\nAll jobs are finished!"<<'\n';
}

/* Implementation of class DataManager  /// /// ///
*/
DataManager::DataManager()
{
	// TODO
}

DataManager::~DataManager()
{
	// TODO
}

DataSet DataManager::LaodData(char *path)
{
	DataSet ds;
	if(!path)
	{
		ds.dim = 1;
		ds.size = 20;
		ds.X = (float*) malloc( sizeof(float)* ds.dim * ds.size );
		ds.Y = (float*) malloc( sizeof(float)* ds.size );
		for(uint i = 0; i < ds.size; ++i)
		{
			ds.X[i] = (float)i/20.0 - 0.5;
			ds.Y[i] = sin(ds.X[i]);//ds.X[i] > 0 ? 0 : 1;
		}
		return ds;
	}

	ifstream file(path);
	assert(file.is_open());

	file >> ds.size >> ds.dim;
	ds.X = (float*) malloc( sizeof(float)* ds.dim * ds.size );
	ds.Y = (float*) malloc( sizeof(float)* ds.size );

	int b = 0;
	for(uint i = 0; i < ds.size; ++i, b += ds.dim)
	{
		file >> ds.Y[i];
		for(uint j = 0; j < ds.dim; ++j)
			file >> ds.X[b+j];
	}
	file.close();
	return ds;
}

void DataManager::DeleteData(DataSet &ds)
{
	// TODO
}

/* Implementation of class ParameterManager  /// /// ///
*/
ParameterManager::ParameterManager(char *path)
{}
ParameterManager::~ParameterManager()
{}


/* Implementation of some test cases  /// /// ///
*/
void test_data()
{
	DataManager dm;
	DataSet ds = dm.LaodData("nn_test_data.txt");

	uint b = 0;
	for(uint i = 0; i < ds.size; ++i, b += ds.dim)
	{
		cout << "Sample " << i << " Y= " << ds.Y[i] << "\n";
		for(uint j = 0; j < ds.dim; ++j)
			cout << ds.X[b+j] << "\t";
		cout << endl;
	}
}