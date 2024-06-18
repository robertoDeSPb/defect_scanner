cmake_minimum_required(VERSION 3.10)
project(cvStudy)

# Укажите путь к OpenCV
set(OpenCV_DIR "D:/Cpp Projects/cv/opencv/build")

# Найдите OpenCV
find_package(OpenCV REQUIRED)

# Добавьте исполняемый файл
add_executable(cvStudy main.cpp)

# Ссылки на библиотеки OpenCV
target_include_directories(cvStudy PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cvStudy ${OpenCV_LIBS})
