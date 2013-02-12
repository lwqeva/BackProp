#pragma once

typedef unsigned int uint;
/* JobManager is in charge of managing modules and jobs.
*/

#define MAX_JOBS	64
class JobManager
{
public:

	class Job
	{
	public :
		Job(void (*func)(void));
		void run();
	private:
		void (*func)(void);
	};

	bool AppendJob(Job *job);
	void Launch();

	static JobManager& GetManager();
	
private:
	JobManager();
	static JobManager *jobmgr;

	Job *jobs[MAX_JOBS];
	int job_in_queue;
};


struct DataSet
{
	float *X;
	float *Y;
	uint size;
	uint dim;
};

class DataManager
{
public:
	DataManager();
	~DataManager();

	DataSet LaodData(char *path=0);
	void DeleteData(DataSet &ds);
};

class ParameterManager
{
public:
	ParameterManager(char *path=0);
	~ParameterManager();
};


void test_data();