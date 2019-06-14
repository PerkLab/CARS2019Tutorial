
@echo OFF

if %1.==. goto No1

set CurrentPath=%cd%
set CurrentDrive=%CD:~0,2%
set EnvironmentPath=%1
set EnvironmentDrive=%~d1

:: Create the environment and activate it

call conda create -y -p %EnvironmentPath% tensorflow-gpu
call activate %EnvironmentPath%

:: Install additional packages

call conda install -y keras pandas opencv==3.4.2 jupyter git scikit-learn matplotlib Pillow ipywidgets

:: Install pyIGTLink from sourceopen

call git clone -b pyIGTLink_client https://github.com/SlicerIGT/pyIGTLink.git %ProjectPath%\pyIGTLink
call pip install -e %ProjectPath%\pyIGTLink

:: Install keras-vis from source

%EnvironmentDrive%
cd %EnvironmentPath%

call git clone https://github.com/raghakot/keras-vis.git %EnvironmentPath%\keras-vis
cd keras-vis
call python setup.py install

%CurrentDrive%
cd %CurrentPath%


:: Exiting install script

GOTO End1

:No1
  echo.
  echo Usage: %~n0 ENVIRONMENT_PATH
  echo E.g.: %~n0 c:\MyProject
  echo.
  echo Note: If admin access is needed to write the environment path, then make sure to start this Anaconda Prompt in Administrator mode.
  echo.
goto End1

:End1