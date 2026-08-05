POLISH - Advanced Wireline Data Preprocessing System
=====================================================
Standalone edition for Windows. Python is NOT required.


HOW TO RUN
----------
1. Extract this entire folder to somewhere you can write to, such as your
   Desktop or Documents. Do not run it from inside the .zip file.

2. Open the extracted folder and run WirelinePreprocessing.exe

3. The first launch takes a little longer than later ones while Windows
   caches the program files. Subsequent launches are faster.

Keep the folder intact. The .exe needs the files next to it and will not run
if it is moved out on its own.


IF WINDOWS WARNS YOU
--------------------
Windows SmartScreen may show "Windows protected your PC" because the program
is not code-signed. Choose "More info", then "Run anyway".

Some antivirus products flag newly built applications they have not seen
before. If the program is quarantined, restore it and add the folder to your
antivirus exclusions.


IF IT DOES NOT START
--------------------
The program writes a diagnostic report when it fails. Open this folder:

    %LOCALAPPDATA%\POLISH\logs

Paste that path into the Windows Explorer address bar. Inside you will find:

    session.log         one line per launch
    crash-<date>.log    written when a failure occurs

Send the most recent crash-*.log file to support.


ABOUT THOSE LOG FILES - PRIVACY
-------------------------------
These files stay on your computer. The program has no networking code and
sends nothing anywhere, ever.

The reports deliberately contain technical information only: Windows version,
the versions of the internal components, and the location in the program where
the failure happened.

They do NOT contain well data, curve values, or LAS or DLIS file contents.
Folders are removed from any path, and the names of your data files are replaced
with just their file type (for example a log becomes <data.las>), so a well,
UWI, field, operator or prospect name held in a folder or file name never
reaches a report. Only technical facts remain, including the location in the
program's own code where the failure happened.

You can read any of these files in Notepad before sending one, and you can
delete them at any time.


SYSTEM REQUIREMENTS
-------------------
Windows 10 or 11, 64-bit
8 GB RAM recommended
About 1 GB of free disk space
