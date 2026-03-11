# libgp - Gaussian process library for Machine Learning
# Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
# All rights reserved.


add_executable(gpdense gp_example_dense.cc)
target_link_libraries(gpdense gp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 Include Directories: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")

# 查找CasADi包
find_package(casadi 3.7.0 REQUIRED)

include_directories(
        include
        ${Python3_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
)

add_executable(2D_example 2D_example.cc)
target_link_libraries(2D_example gp ${Python3_LIBRARIES})

add_executable(toy_example toy_example.cc)
target_link_libraries(toy_example gp ${Python3_LIBRARIES})

add_executable(periodic_example periodic_example.cc)
target_link_libraries(periodic_example gp ${Python3_LIBRARIES} casadi::casadi)

# add_executable(recursive_gp_example recursive_gp_example.cc)
# target_link_libraries(recursive_gp_example gp ${Python3_LIBRARIES})

# add_executable(usv_gp_example usv_gp_example.cc)
# target_link_libraries(usv_gp_example gp ${Python3_LIBRARIES})

# add_executable(usvResidualDynamicsGP usvResidualDynamicsGP.cc)
# target_link_libraries(usvResidualDynamicsGP gp ${Python3_LIBRARIES})

add_executable(cppad_example cppad_example.cc)
target_link_libraries(cppad_example gp ${Python3_LIBRARIES})

add_executable(casadi_example casadi_example.cc)
target_link_libraries(casadi_example gp ${Python3_LIBRARIES} casadi::casadi)

add_executable(maneuvering_example_KVLCC2 maneuvering_example_KVLCC2.cc)
target_link_libraries(maneuvering_example_KVLCC2 gp ${Python3_LIBRARIES} casadi::casadi)
