# Lightbox NERF
Handles data repackaging, training, and rendering for Lightbox NERFs; expected to be used with EC2 instances

## SageMaker startup

Create new notebook instance in SageMaker with the following options:

| Property                | Value                                            | 
| ---                     | ---                                              |
| Notebook instance type  | ?ml.p2.xlarge                                    |
| Lifecycle configuration | ?lifecycle-setup-tflow-nerf-on-start-only        |
| Volume size in GB       | Can be anything, but this gets billed as storage | 
| Git repository          | lbx-nerf - GitHub                                |

Any options not listed above should be left at their default values.

Once you've started the instance, create a new terminal, navigate to this repo, and type
~~~
$ pip install pipenv
~~~
to install the pipenv package manager (similar to anaconda). Now type
~~~
$ pipenv shell
~~~
to automatically load the environment stored in the .venv folder.

## Notes / Learnings
### pipenv
When you create a new environment with pipenv, the default behavior is to store the packages in `~/.local/share/virtualenvs/`. This is bad because it uses up the limited space outside the SageMaker partition and it will also get wiped every time you stop your EC2 instance.

Luckily, there is a workaround. If you create an empty folder called ".venv" in the top level of your project directory, pipenv will use that folder to store the environment created for your project.