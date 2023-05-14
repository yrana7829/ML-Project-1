ML Project-1 :- Student performance analysis

Project Description :- A Regression analysis project

Problem Statement :-
To analyse the student performances based on the total marks scored by them and various variables including categorical and numeric variables

Project Structure:-
All components are stored in the src--> components folder and all the pipelines used during the project are stored in src--> pipelines folder.
logger.py and exception.py files inside src are being used in various places in project in order to log the info and raise the custom exceptions.

exception.py
The code will handle any exception occuring in any component of the project. Whenever there is an error we need to return it as a proper message.

logger.py
The logger will be used to log the neccesary information on various stage of project helping in debug. Any exception aoccuring at any moment will be logged in to a file.
