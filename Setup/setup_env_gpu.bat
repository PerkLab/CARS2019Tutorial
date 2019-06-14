
@echo OFF

if %1.==. goto No1

set EnvironmentPath=%1

:: Create the environment and activate it

call conda create -y -p %EnvironmentPath% tensorflow-gpu
call activate %EnvironmentPath%

:: Install additional packages

call conda install -y keras pandas opencv==3.4.2 jupyter git scikit-learn matplotlib Pillow ipywidgets

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