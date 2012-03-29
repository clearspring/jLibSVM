export OMP_NUM_THREADS=${1}
time ./svm-train -c 16 -g 4 -m 400 ijcnn1
