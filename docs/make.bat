@ECHO OFF
REM docs/make.bat
REM Command file for Sphinx documentation on Windows

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

if "%1" == "help" (
	:help
	%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
	goto end
)

if "%1" == "clean" (
	echo.Removing build directory...
	if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
	goto end
)

if "%1" == "livehtml" (
	echo.Starting live reload server...
	sphinx-autobuild "%SOURCEDIR%" "%BUILDDIR%\html" --host 0.0.0.0 --port 8000
	goto end
)

if "%1" == "linkcheck" (
	echo.Checking external links...
	%SPHINXBUILD% -b linkcheck "%SOURCEDIR%" "%BUILDDIR%\linkcheck" %SPHINXOPTS% %O%
	goto end
)

if "%1" == "coverage" (
	echo.Checking documentation coverage...
	%SPHINXBUILD% -b coverage "%SOURCEDIR%" "%BUILDDIR%\coverage" %SPHINXOPTS% %O%
	type "%BUILDDIR%\coverage\c.txt"
	goto end
)

if "%1" == "spelling" (
	echo.Checking spelling...
	%SPHINXBUILD% -b spelling "%SOURCEDIR%" "%BUILDDIR%\spelling" %SPHINXOPTS% %O%
	goto end
)

if "%1" == "all" (
	echo.Building all formats...
	%SPHINXBUILD% -M html "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	%SPHINXBUILD% -M epub "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	%SPHINXBUILD% -M pdf "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	goto end
)

if "%1" == "rtd" (
	echo.Building documentation for Read the Docs...
	%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html" -W --keep-going %SPHINXOPTS% %O%
	goto end
)

if "%1" == "quick" (
	echo.Quick build ^(no API docs regeneration^)...
	%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html" -E %SPHINXOPTS% %O%
	goto end
)

if "%1" == "apidoc" (
	echo.Generating API documentation...
	sphinx-apidoc -o source\api ..\src\pybuildingcluster --force --module-first
	goto end
)

if "%1" == "rebuild" (
	echo.Full rebuild with API docs...
	if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
	sphinx-apidoc -o source\api ..\src\pybuildingcluster --force --module-first
	%SPHINXBUILD% -M html "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	goto end
)

if "%1" == "strict" (
	echo.Building with strict warnings...
	%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html" -W -n --keep-going %SPHINXOPTS% %O%
	goto end
)

REM Default case - pass through to sphinx-build
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd