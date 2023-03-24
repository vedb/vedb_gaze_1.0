export baseDir=~/dev/bates
export projectDir=~/dev/bates/vedb_run
export envName="vedb_analysis"
export ancondaLoc="/opt/anaconda3/etc/profile.d/conda.sh"

# Check for interactive environment
set -e

if [[ $- == *i* ]]
then
	echo 'Interactive mode detected.'
else
	echo 'Not in interactive mode! Please run with `bash -i set_up_vedb_analysis_environment.sh`'
	exit
fi

# Make source code directory for git libraries
if [ -d $projectDir ]
	then
		echo "Code directory found."
	else
		echo "Creating $projectDir directory."
		mkdir $projectDir
fi

# Environment setup
# Assure mamba is installed
if hash mamba 2>/dev/null; then
	echo ">>> Found mamba installed."
	# mamba update mamba
else
	conda install mamba -n base -c conda-forge
	# mamba update mamba
fi

# Create initial environment
if (conda env list | grep $envName); then
	echo "Found environment $envName"
else
	echo "** Creating Conda Environment: $envName"
	mamba env create -f $projectDir/env_setup/environment_$envName.yml
    echo "** Mamba env $envName created"
fi;

# Activate it
# Use your location here...
source $ancondaLoc
conda activate $envName

# Assure g++ is installed
if [ `which g++` ]; then
	echo ">>> g++ found"
else
	echo ">>> installing g++ ..."
	sudo apt-get install g++
fi

# Load list of libraries to install from which repos; should be mostly (all) vedb
# Retrieve and install libraries from git
# Need at least bash version 4.0
declare -A git_repos
git_repos['py-thin-plate-spline']='git@github.com:cheind/py-thin-plate-spline.git'
# git_repos['vedb-gaze']='git@github.com:vedb/vedb-gaze.git'
# git_repos["file_io"]="git@github.com:vedb/file_io.git"
# git_repos['plot_utils']='git@github.com:piecesofmindlab/plot_utils.git'
# git_repos['pupil-detectors-ml']='git@github.com:marklescroart/pupil-detectors.git pupil-detectors-ml'
# git_repos['vedb-odometry']='git@github.com:vedb/vedb-odometry.git'

for repo in "${!git_repos[@]}"
do
	echo "** Installing: $repo"
    cd $projectDir
	if [ -d $repo ]; then
		:
	else
		git clone ${git_repos[$repo]}
	fi
	cd $repo
	python setup.py install
	cd $projectDir
done
