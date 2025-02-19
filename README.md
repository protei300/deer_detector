Сборка исполняемого файла для платформы Windows
-----------------------------------------------

В процессе сборки приложения настоятельно рекомендуется использовать
виртуальные окружения для python:

```shell
python3 -m venv venv
venv/Scripts/activate.bat
python3 -m pip install -r requirements.txt
```

После создания виртуального окружения можно приступать непосредственно к сборке.
Сборка приложения для операционных систем Windows осуществляется с помощью
утилиты `pyinstaller`.
Обратите внимание на то, что в проекте используется специфическая версия pyinstaller.
Это связано с ошибкой в старших версиях, которая не позволяет собрать исполняемый файл
на некоторых версиях операционной системы.

Сборка исполняемого файла проста:

```shell
pyinstaller.exe --clean --windowed --add-data="main.ui;." --add-data="Models/;Models/" .\main.py
```
После выполнения этой команды в директории `dist` появится каталог `main`, внутри которого
будут находиться все необходимые для запуска программы зависимости. В этой же директории будет
лежать файл `main.exe` который и нужно запускать.
