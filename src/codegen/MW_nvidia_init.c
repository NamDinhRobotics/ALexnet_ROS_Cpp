/* Copyright 2020 The MathWorks, Inc. */
#include <stdio.h>
#include <stdlib.h>
#include "MW_nvidia_init.h"

// Overrun detection function
void reportOverrun(int taskId)
{
#ifdef MW_NVIDIA_DETECTOVERRUN
    printf("Overrun detected: The sample time for the rate %d is too short.\n", taskId);
    fflush(stdout);
#endif
}
