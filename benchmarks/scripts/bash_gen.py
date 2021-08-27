import os


def execute_on_server(bash_location, bash_name, script_location, arguments,
                      python_path="/home/icb/johannes.ostner/anaconda3/bin/python"):
    """
    Script to make a bash script that pushes a job to the ICB CPU servers where another python script is executed.

    Parameters
    ----------
    bash_location: str
        path to the folder where bash file, out and error files are written
    bash_name: str
        name of your job. The bash file, out and error files will have this name (with respective suffixes)
    script_location: str
        path to the python script
    arguments: dict
        list of arguments that is passed to the script

    Returns
    -------

    """

    # Build bash file
    bash_file = bash_location + bash_name + "_script.sh"
    with open(bash_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH -o {bash_location}{bash_name}_out.o\n")
        fh.writelines(f"#SBATCH -e {bash_location}{bash_name}_error.e\n")
        fh.writelines("#SBATCH -p cpu_p\n")
        fh.writelines("#SBATCH --constraint='avx'\n")
        fh.writelines("#SBATCH -c 1\n")
        fh.writelines("#SBATCH --mem=32000\n")
        fh.writelines("#SBATCH --nice=10000\n")
        fh.writelines("#SBATCH -t 3-00:00:00\n")

        execute_line = f"{python_path} {script_location} "
        for key, value in arguments.items():
            if isinstance(value, list):
                execute_line = execute_line + f"--{key} {' '.join(str(v) for v in value)} "
            else:
                execute_line = execute_line + f"--{key} {str(value)} "
        fh.writelines(execute_line)

    # Run the bash file you just generated
    os.system(f"sbatch {bash_file}")
    # os.system(f"source {bash_file}")

