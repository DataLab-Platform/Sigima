Setting up Development Environment
==================================

Getting started with Sigima development is easy.

Here is what you will need:

1. An integrated development environment (IDE) for Python. We recommend
   [Spyder](https://www.spyder-ide.org/) or [Visual Studio Code](https://code.visualstudio.com/),
   but any IDE will do.

2. A Python distribution. We recommend [WinPython](https://winpython.github.io/),
   on Windows, or [Anaconda](https://www.anaconda.com/), on Linux or Mac.
   But, again, any Python distribution will do.

3. A clean project structure (see below).

4. Test data (see below).

5. Environment variables (see below).

6. Third-party software (see below).

Development Environment
-----------------------

If you are using [Spyder](https://www.spyder-ide.org/), thank you for supporting
the scientific open-source Python community!

If you are using Visual Studio Code, that's also an excellent choice (for other
reasons). We recommend installing the following extensions:

| Extension | Description |
| --------- | ----------- |
| [gettext](https://marketplace.visualstudio.com/items?itemName=mrorz.language-gettext) | Gettext syntax highlighting |
| [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) | Python language server |
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | Python extension |
| [reStructuredText Syntax highlighting](https://marketplace.visualstudio.com/items?itemName=trond-snekvik.simple-rst) | reStructuredText syntax highlighting |
| [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) | Extremely fast Python linter and code formatter |
| [Todo Tree](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree) | Todo tree |
| [Insert GUID](https://marketplace.visualstudio.com/items?itemName=heaths.vscode-guid) | Insert GUID |
| [XML Tools](https://marketplace.visualstudio.com/items?itemName=DotJoshJohnson.xml) | XML Tools |

Python Environment
------------------

Sigima requires the following :

* Python (e.g. WinPython)

* Additional Python packages

Installing all required packages :

    pip install --upgrade -r requirements.txt

If you are using [WinPython](https://winpython.github.io/), thank you for supporting
the scientific open-source Python community!

The following table lists the currently officially used Python distributions:

| Python version | Status       | WinPython version |
| -------------- | ------------ | ----------------- |
| 3.10           | OK           | 3.10.11.1         |
| 3.11           | OK           | 3.11.5.0          |
| 3.12           | OK           | 3.12.3.0          |
| 3.13           | OK           | 3.13.2.0          |
| 3.14           | OK           | 3.14.5.0          |

⚠ We strongly recommend using the `.dot` versions of WinPython which are lightweight
and can be customized to your needs (using `pip install -r requirements.txt`).

✅ We also recommend using a dedicated WinPython instance for Sigima.

Test data
---------

Sigima test data are located in different folders, depending on their nature or origin.

Required data for unit tests are located in "sigima\data\tests" (public data).

A second folder %SIGIMA_DATA% (optional) may be defined for additional tests which are
still under development (or for confidential data).

Specific environment variables
------------------------------

Visual Studio Code configuration used in `launch.json` and/or `tasks.json`
(examples) :

    @REM Folder containing additional working test data
    set SIGIMA_DATA=C:\Dev\Projets\SIGIMA_data

Visual Studio Code `.env` file:

* This file is used to set environment variables for the application.
* It is used to set the `PYTHONPATH` environment variable to the root of the project.
* This is required to be able to import the project modules from within VS Code.
* To create this file, copy the `.env.template` file to `.env`
  (and eventually add your own paths).

Third-party Software
--------------------

The following software may be required for maintaining the project:

| Software | Description |
| -------- | ----------- |
| [gettext](https://mlocati.github.io/articles/gettext-iconv-windows.html) | Translations |
| [Git](https://git-scm.com/) | Version control system |
| [ImageMagick](https://imagemagick.org/) | Image manipulation utilities |
