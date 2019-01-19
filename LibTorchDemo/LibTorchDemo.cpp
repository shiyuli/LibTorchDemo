// LibTorchDemo.cpp : Defines the entry point for the application.
//

#include "LibTorchDemo.h"

#define GPU_SUPPORT false

int main()
{
	ModelManager *model_manager = new ModelManager("model.pt", GPU_SUPPORT);
	model_manager->load_model();
	model_manager->exec_model();

	system("pause");
	return 0;
}
