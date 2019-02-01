TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    functions.cpp

INCLUDEPATH += /usr/local/include

LIBS += /usr/local/lib/libopencv_*.so

HEADERS += \
    functions.h
