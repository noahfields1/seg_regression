#!/bin/bash

# Name of your job
# CHANGE THIS JOB NAME
#SBATCH --job-name=mediumSteady
#SBATCH --partition=compute

# Specify the name of the output file. The %j specifies the job ID
# CHANGE THIS JOB OUTPUT FILE NAME
#SBATCH --output=log.o%j

# Specify a name of the error file. The %j specifies the job ID
# CHANGE THIS JOB ERROR FILE NAME
#SBATCH --error=log.e%j

# The walltime you require for your simulation
#SBATCH --time=3:00:00

# Number of nodes you are requesting for your job. You can have 24 processors per node, so plan accordingly
# CAN CHANGE THE NUMBER OF NODES
# 20,000 elements/processor is a good rule of thumb for scaling for FSI. (i.e. 4million FSI mesh is about 192 processors = 8 nodes)
# 40,0000 elements/processor for rigid (i.e. 4million rigid mesh is about 96 processors = 4 nodes)
#SBATCH --nodes=1

# Number of processors per node
#SBATCH --ntasks-per-node=24

# Send an email to this address when you job starts and finishes
# CHANGE THIS EMAIL ADDRESS
#SBATCH --mail-user=contact.gabriel.maher@gmail.com
#SBATCH --mail-type=all

# Modules
module purge
module load gnu/7.2.0
module load lapack
#module load gnu openmpi_ib
module load mvapich2_ib/2.3.2
module load python
module load scipy
module load boost


# Name of the executable you want to run
/home/gdmaher/svSolver/svpre.exe model_sim.svpre
ibrun /home/gdmaher/svSolver/svsolver-mpich.exe
/home/gdmaher/svSolver/svpost.exe -indir 24-procs_case -outdir . -start 300 -stop 400 -incr 10 -vtu all_results.vtu -vtp all_results.vtp -vtkcombo -all
rm -rf 24-procs_case
