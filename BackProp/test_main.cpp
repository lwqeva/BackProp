#include <iostream>
#include "JobManager.h"
#include "BackProp.h"
using namespace BackProp;

int main()
{
	JobManager &mgr = JobManager::GetManager();

	mgr.AppendJob(new JobManager::Job(test_initial_weight));
	mgr.Launch();

	system("pause");
	return 0;
}
